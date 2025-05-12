# runs in jupyter container 
import os
import numpy as np
import tarfile
import shutil
import re
import unicodedata
from tqdm import tqdm
from multiprocessing import Pool
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import ARRAY, TEXT
import psycopg2
import sys

METADATA_TABLE = sys.argv[1] #"arxiv_metadata"
CHUNKS_TABLE   = sys.argv[2] #"arxiv_chunks"

engine = create_engine(
    'postgresql+psycopg2://rg5073:rg5073pass@129.114.27.112:5432/cleaned_meta_data_db',
    pool_size=12,
    max_overflow=0,
    pool_timeout=30,
)

create_metadata_sql = f"""
CREATE TABLE IF NOT EXISTS {METADATA_TABLE} (
    id TEXT,
    submitter TEXT,
    authors TEXT,
    title TEXT,
    comments TEXT,
    "journal-ref" TEXT,
    doi TEXT,
    "report-no" TEXT,
    categories TEXT,
    abstract TEXT,
    latest_version TEXT,
    txt_filename TEXT,
    created_yymm TEXT
);
"""

create_chunks_sql = f"""
CREATE TABLE IF NOT EXISTS {CHUNKS_TABLE} (
    paper_id TEXT,
    chunk_id INT,
    txt_filename TEXT,
    query TEXT,
    chunk_data TEXT
);
"""

with engine.begin() as conn:
    conn.execute(text(create_metadata_sql))
    conn.execute(text(create_chunks_sql))

with engine.connect() as conn:
    row_count = conn.execute(text(f"SELECT COUNT(*) FROM {METADATA_TABLE};")).scalar()

csv_path = os.path.join('/req-data', 'arxiv_cleaned_v1.csv')

if row_count == 0:
    df = pd.read_csv(csv_path)
    print("Our metadata contains:", len(df), "records")

    raw_conn = engine.raw_connection()
    cur = raw_conn.cursor()
    with open(csv_path, 'r') as f:
        cur.copy_expert(f"""
            COPY {METADATA_TABLE}
            FROM STDIN
            WITH CSV HEADER
        """, f)
    raw_conn.commit()
    cur.close()
    raw_conn.close()
    print("Data loaded into", METADATA_TABLE)
else:
    print(f"Table {METADATA_TABLE} is not empty — skipping CSV load.")

workspace_dir = '/data'
text_files_data_path = '/req-data'
# tar_files_list = os.listdir(text_files_data_path)
# new — only .tar files
tar_files_list = [
    fn for fn in os.listdir(text_files_data_path)
    if fn.lower().endswith('.tar')
]

print("Text tar files list:", tar_files_list)

def load_existing_txt_filenames():
    query = text(f"SELECT txt_filename FROM {METADATA_TABLE};")
    with engine.connect() as conn:
        return {row[0] for row in conn.execute(query).fetchall()}

existing_txt = load_existing_txt_filenames()
print("Total text files to process:", len(existing_txt))

def chunk_text(text, chunk_size_words=650):
    words = text.split()
    return [" ".join(words[i:i+chunk_size_words]) for i in range(0, len(words), chunk_size_words)]

def simple_clean_text_remove_references(text):
    # cleaning steps...
    text = text.replace('\x00', '')
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r'\$.*?\$', ' ', text)
    text = re.sub(r'\\[a-zA-Z]+\{.*?\}', ' ', text)
    text = re.sub(r'[\u2200-\u22FF\u2300-\u23FF]', ' ', text)
    text = re.sub(r'⟨.*?⟩', ' ', text)
    text = re.sub(r'\[\d+(,\s*\d+)*\]', ' ', text)
    text = re.sub(r'\(\d+(\.\d+)?\)', ' ', text)
    text = re.sub(r'\{.*?\}', ' ', text)
    text = re.sub(r'([A-Za-z0-9]+\s*[=<>]\s*[A-Za-z0-9^+\-*/\s]+)', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\?\!]', ' ', text)
    text = re.sub(r'-\n\s*', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.split(r'\bReferences\b', text, flags=re.IGNORECASE)[0]
    return text.strip()

def process_single_file(task):
    file_path, txt_filename = task
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw = f.read()
        clean = simple_clean_text_remove_references(raw)
        chunks = chunk_text(clean)
        paper_id = txt_filename.replace(".txt", "")
        return (paper_id, txt_filename, chunks)
    except Exception as e:
        print(f"Chunking failed for {txt_filename}: {e}")
        return None

def insert_chunks(conn, entries):
    insert_stmt = text(f"""
        INSERT INTO {CHUNKS_TABLE} (paper_id, chunk_id, txt_filename, query, chunk_data)
        VALUES (:paper_id, :chunk_id, :txt_filename, :query, :chunk_data)
    """)
    batch_size = 500
    inserts = []
    for paper_id, fn, chunks in entries:
        for idx, chunk in enumerate(chunks, start=1):
            inserts.append({
                "paper_id": paper_id,
                "chunk_id": idx,
                "txt_filename": fn,
                "query": "",
                "chunk_data": chunk
            })
    for i in tqdm(range(0, len(inserts), batch_size), desc="Inserting into chunks"):
        conn.execute(insert_stmt, inserts[i : i + batch_size])

def process_tar_file(tar_fn):
    tar_path = os.path.join(text_files_data_path, tar_fn)
    print(f"\n📦 Processing tar: {tar_fn}…")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=workspace_dir)
    folder = tar_fn.replace(".tar", "")
    path = os.path.join(workspace_dir, folder)
    txts = [f for f in os.listdir(path) if f.endswith(".txt") and f in existing_txt]
    tasks = [(os.path.join(path, f), f) for f in txts]
    if not tasks:
        print("No matching .txt files in", tar_fn)
        shutil.rmtree(path)
        return
    with Pool(processes=8) as pool:
        results = list(tqdm(pool.imap_unordered(process_single_file, tasks),
                            total=len(tasks), desc=f"Chunking {tar_fn}", dynamic_ncols=True))
    entries = [r for r in results if r]
    with engine.begin() as conn:
        insert_chunks(conn, entries)
    shutil.rmtree(path)
    print(f"Done {tar_fn}: inserted {len(entries)} papers.")

for tar in tar_files_list:
    process_tar_file(tar)

print("All tar files processed!")
