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
    'postgresql+psycopg2://rg5073:rg5073pass@129.114.27.112:5432/cleaned_meta_data_db',
    pool_size=12,
    max_overflow=0,
    pool_timeout=30,
)

create_sql = """
CREATE TABLE IF NOT EXISTS staging_data_queries (
    query_id INT,
    query TEXT,
    summary_generated TEXT,
    top_k_papers TEXT[],
    user_rating TEXT
);
"""
with engine.begin() as conn:
    conn.execute(text(create_sql))

with open("queries.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()][:50]

# Prepare the INSERT statement
insert_sql = text("""
    INSERT INTO staging_data_queries
      (query_id, query, summary_generated, top_k_papers, user_rating)
    VALUES
      (:qid, :qry, NULL, ARRAY[]::text[], NULL)
""")

# Execute within a transaction
with engine.begin() as conn:
    for idx, qry in enumerate(lines, start=1):
        conn.execute(insert_sql, {"qid": idx, "qry": qry})

create_sql = """
CREATE TABLE IF NOT EXISTS production_data_queries (
    query_id INT,
    query TEXT,
    summary_generated TEXT,
    top_k_papers TEXT[],
    user_rating TEXT
);
"""
with engine.begin() as conn:
    conn.execute(text(create_sql))

with open("queries.txt", "r", encoding="utf-8") as f:
    all_lines = [line.strip() for line in f if line.strip()]

# Take the next 50 (indices 50–99 in zero-based Python)
batch = all_lines[50:100]

insert_sql = text("""
    INSERT INTO production_data_queries
      (query_id, query, summary_generated, top_k_papers, user_rating)
    VALUES
      (:qid, :qry, NULL, ARRAY[]::text[], NULL)
""")

with engine.begin() as conn:
    for offset, qry in enumerate(batch, start=51):
        conn.execute(insert_sql, {"qid": offset - 50, "qry": qry})