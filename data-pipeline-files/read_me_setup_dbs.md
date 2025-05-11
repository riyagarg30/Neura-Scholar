docker compose -f docker-compose-meta-data-db-jupyter.yaml up -d

#first time setup:
 docker exec -it cleaned_meta_data_postgres bash
 psql -U rg5073 -d cleaned_meta_data_db

 CREATE TABLE arxiv_metadata (
    id TEXT,
    submitter TEXT,
    authors TEXT,
    title TEXT,
    comments TEXT,
    journal-ref TEXT,
    doi TEXT,
    report-no TEXT,
    categories TEXT,
    abstract TEXT,
    latest_version TEXT,
    txt_filename TEXT,
    created_yymm TEXT
);

CREATE TABLE arxiv_chunks (
    paper_id TEXT,
    chunk_id INT,
    txt_filename TEXT,
    query TEXT,
    chunk_data TEXT
);

CREATE TABLE arxiv_chunks_eval_2 (
    paper_id TEXT,
    chunk_id TEXT,
    txt_filename TEXT,
    query TEXT,
    chunk_data TEXT,
    paper_cited TEXT[]
);

\copy arxiv_metadata(
    id, submitter, authors, title, comments, journal_ref, doi, report_no,
    categories, license, abstract, update_date, authors_parsed, latest_version, latest_created, txt_filename
)
FROM '/csvfiles/arxiv_cleaned_v3.csv'
WITH (FORMAT csv, HEADER true);



