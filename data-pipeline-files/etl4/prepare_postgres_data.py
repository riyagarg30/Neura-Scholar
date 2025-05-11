# runs in jupyter container 
import os
import numpy as np
from sqlalchemy import create_engine, text
import pandas as pd
import tarfile
import shutil
import re
import unicodedata
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import ARRAY, TEXT
from tqdm import tqdm
from multiprocessing import Pool
import psycopg2
from sqlalchemy import inspect

engine = create_engine(
    'postgresql+psycopg2://rg5073:rg5073pass@129.114.27.112:5432/meta_data_chunks',
    pool_size=12,
    max_overflow=0,
    pool_timeout=30,
)

create_sql = """
CREATE TABLE IF NOT EXISTS arxiv_metadata (
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

with engine.begin() as conn:
    conn.execute(text(create_sql))

create_chunks_table_sql = """
CREATE TABLE IF NOT EXISTS arxiv_chunks (
    paper_id TEXT,
    chunk_id INT,
    txt_filename TEXT,
    query TEXT,
    chunk_data TEXT
);
"""

with engine.begin() as conn:
    conn.execute(text(create_chunks_table_sql))

csv_path = os.path.join('/data-obj/meta-data', 'arxiv_cleaned_v1.csv')

df = pd.read_csv(csv_path)
print("Our metadata contains: ",len(df), " records")

raw_conn = engine.raw_connection()
cur = raw_conn.cursor()

with open(csv_path, 'r') as f:
    cur.copy_expert(f"""
        COPY arxiv_metadata
        FROM STDIN
        WITH CSV HEADER
    """, f)

raw_conn.commit()
cur.close()

workspace_dir = '/data'
text_files_data_path = os.path.join('/data-obj', "text-files-data")
tar_files_list = os.listdir(text_files_data_path)
print("Text tar files list",tar_files_list)

total_files = 0
fail_count = 0

def load_existing_txt_filenames():
    query = text("SELECT txt_filename FROM arxiv_metadata;")
    with engine.connect() as conn:
        result = conn.execute(query)
        pdf_filenames = {row[0] for row in result.fetchall()}
    return pdf_filenames
existing_pdf_filenames = load_existing_txt_filenames()
print("Total text files to process:", len(existing_pdf_filenames))

# Chunking function
def chunk_text(text, chunk_size_words=650):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size_words):
        chunks.append(" ".join(words[i:i+chunk_size_words]))
    return chunks

def simple_clean_text_remove_references(text):
    # 1. Remove null bytes
    text = text.replace('\x00', '')

    # 2. Normalize Unicode (like ﬁ → fi)
    text = unicodedata.normalize("NFKD", text)

    # 3. Remove inline LaTeX/math ($...$) and LaTeX commands (\command{...})
    text = re.sub(r'\$.*?\$', ' ', text)                 # Remove math in $
    text = re.sub(r'\\[a-zA-Z]+\{.*?\}', ' ', text)       # Remove \commands{...}
    
    # 4. Remove Unicode math symbols and special symbols (sets, operators, etc)
    text = re.sub(r'[\u2200-\u22FF\u2300-\u23FF]', ' ', text)

    # 5. Remove anything between ⟨...⟩ (angle brackets)
    text = re.sub(r'⟨.*?⟩', ' ', text)

    # 6. Remove references like [1], [2,5,10]
    text = re.sub(r'\[\d+(,\s*\d+)*\]', ' ', text)

    # 7. Remove numbered equations like (123), (4.5)
    text = re.sub(r'\(\d+(\.\d+)?\)', ' ', text)

    # 8. Remove any remaining weird LaTeX leftovers like {some text}
    text = re.sub(r'\{.*?\}', ' ', text)

    # 9. Remove equations written like "E = mc^2" (detect common formula style)
    text = re.sub(r'([A-Za-z0-9]+\s*[=<>]\s*[A-Za-z0-9^+\-*/\s]+)', ' ', text)

    # 10. Remove all non-ASCII except basic punctuations
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # 11. Remove any special characters except basic word characters and sentence punctuation
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\?\!]', ' ', text)

    # 12. Remove extra hyphenated line breaks
    text = re.sub(r'-\n\s*', '', text)

    # 13. Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # 14. Remove the References section completely
    text = re.split(r'\bReferences\b', text, flags=re.IGNORECASE)[0]

    # 15. Final strip
    return text.strip()

# Per-file processing (for parallel chunking)
def process_single_file(task):
    file_path, txt_filename = task
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text_content = f.read()
        cleaned_text = simple_clean_text_remove_references(text_content)
        chunks = chunk_text(cleaned_text)
        paper_id = txt_filename.replace(".txt", "")
        return (paper_id, txt_filename, chunks)
    except Exception as e:
        print(f"Chunking failed for {txt_filename}: {e}")
        return None
    
# ⭐ Parallel INSERT into arxiv_chunks
def insert_chunks(conn, entries):
    """
    entries: list of (paper_id, txt_filename, chunks_list)
    """
    inserts = []
    for paper_id, txt_filename, chunks in entries:
        for idx, chunk in enumerate(chunks, start=1):
            inserts.append({
                "paper_id": paper_id,
                "chunk_id": idx,
                "txt_filename": txt_filename,
                "query": "",  # optional: you can populate later
                "chunk_data": chunk
            })

    if not inserts:
        return

    insert_stmt = text("""
        INSERT INTO arxiv_chunks (paper_id, chunk_id, txt_filename, query, chunk_data)
        VALUES (:paper_id, :chunk_id, :txt_filename, :query, :chunk_data)
    """)

    batch_size = 500  # Adjust based on memory
    for i in tqdm(range(0, len(inserts), batch_size), desc="Inserting into arxiv_chunks"):
        batch = inserts[i:i+batch_size]
        conn.execute(insert_stmt, batch)

# ⭐ Per-tar processing
def process_tar_file(tar_filename):
    global total_files, fail_count

    tar_path = os.path.join(text_files_data_path, tar_filename)

    if not os.path.exists(tar_path):
        print(f"Tar file not found: {tar_path}")
        return

    print(f"\n📦 Processing tar: {tar_filename}...")

    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=workspace_dir)

    extracted_folder_name = tar_filename.replace(".tar", "")
    extracted_folder_path = os.path.join(workspace_dir, extracted_folder_name)

    if not os.path.exists(extracted_folder_path):
        print(f"Extracted folder missing: {extracted_folder_path}")
        return

    print(f"🔍 Extracted to: {extracted_folder_path}")

    txt_files_list = os.listdir(extracted_folder_path)
    print(f"📄 Found {len(txt_files_list)} text files.")

    tasks = []
    for filename in txt_files_list:
        if filename.endswith(".txt") and filename in existing_pdf_filenames:
            file_path = os.path.join(extracted_folder_path, filename)
            tasks.append((file_path, filename))

    if not tasks:
        print(f"No matching text files found in {tar_filename}")
        shutil.rmtree(extracted_folder_path)
        return

    # Process all files in parallel
    with Pool(processes=8) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_file, tasks),
            total=len(tasks),
            desc=f"Chunking {tar_filename}",
            dynamic_ncols=True
        ))

    processed_entries = [r for r in results if r is not None]

    print(f"Processed {len(processed_entries)} / {len(tasks)} files ready for DB insert.")

    with engine.begin() as conn:
        try:
            insert_chunks(conn, processed_entries)
            total_files += len(processed_entries)
        except Exception as e:
            fail_count += len(processed_entries)
            print(f" Insert failed for {tar_filename}: {e}")

    shutil.rmtree(extracted_folder_path)
    print(f"Deleted extracted folder: {extracted_folder_path}")

for tar_filename in tar_files_list:
    process_tar_file(tar_filename)

print("All tar files processed!")
print(f"Total papers inserted: {total_files}")
print(f"Total failed inserts: {fail_count}")