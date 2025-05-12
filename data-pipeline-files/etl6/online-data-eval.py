#!/usr/bin/env python3
import sys
import time
import random
from sqlalchemy import create_engine, text, inspect

DB_URL        = 'postgresql+psycopg2://rg5073:rg5073pass@129.114.27.112:5432/cleaned_meta_data_db'
DELAY_SECONDS = 5 * 6
TOP_K_COUNT   = 5

TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS {table_name} (
    query_id          INT,
    query             TEXT,
    summary_generated TEXT,
    top_k_papers      TEXT[],
    user_rating       TEXT
);
"""

# MOCK BACKEND

def mock_backend(query: str, env: str) -> dict:
    time.sleep(0.1)
    return {
        "summary_generated": f"Mock summary for query {query!r} in '{env}' env.",
        "top_k_papers":     [f"mock_paper_{i}" for i in range(1, TOP_K_COUNT + 1)],
        "user_rating":      random.choice([None, "good", "average", "poor"])
    }

# ARG PARSING

if len(sys.argv) != 2 or sys.argv[1] not in ("staging", "production"):
    print(f"Usage: {sys.argv[0]} <staging|production>")
    sys.exit(1)

env        = sys.argv[1]
table_name = f"{env}_data_queries_online"


# echo=True will print every SQL statement + bound params
engine = create_engine(DB_URL, echo=True, pool_size=12, max_overflow=0)
inspector = inspect(engine)

# List all tables to verify ours is created
print("Existing tables before create:", inspector.get_table_names())

# ENSURE TABLE

with engine.begin() as conn:
    conn.execute(text(TABLE_SCHEMA.format(table_name=table_name)))
    print(f" Ensured table '{table_name}' exists.")

print("Existing tables after create:", inspector.get_table_names())

# RETURNING to rowcount = 1 if success
insert_sql = text(f"""
    INSERT INTO {table_name}
      (query_id, query, summary_generated, top_k_papers, user_rating)
    VALUES
      (:qid, :qry, :summary, :top_k, :rating)
    RETURNING 1
""")

# LOAD QUERIES

with open("queries.txt", "r", encoding="utf-8") as f:
    all_queries = [l.strip() for l in f if l.strip()]

queries = all_queries[:50] if env == "staging" else all_queries[50:100]
print(f"[{env.upper()}] Will process {len(queries)} rows.")

# LOOP & DEBUG INSERT

for idx, qry in enumerate(queries, start=1):
    print(f"\n[{env.upper()}] #{idx} → {qry!r}")
    data = mock_backend(qry, env)

    # one transaction per insert
    with engine.begin() as conn:
        result = conn.execute(
            insert_sql,
            {
                "qid": idx,
                "qry": qry,
                "summary": data["summary_generated"],
                "top_k": data["top_k_papers"],
                "rating": data["user_rating"],
            }
        )

    print("Inserted rowcount:", result.rowcount)

    if idx < len(queries):
        print(f"Sleeping {DELAY_SECONDS} seconds…")
        time.sleep(DELAY_SECONDS)

