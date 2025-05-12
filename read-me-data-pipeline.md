## DataPipeline

A modular ETL framework for processing a large corpus of PDF papers, preparing data for model training, and providing an interactive Streamlit dashboard for inspection.

### Table of Contents

1. [Overview](#overview)
2. [Folder Splits](#folder-splits)
3. [ETL Pipelines](#etl-pipelines)
4. [Streamlit Dashboard](#streamlit-dashboard)
5. [Getting Started](#getting-started)

---

### Overview

This project orchestrates the end-to-end data flow from raw PDF files to ready-to-use training, evaluation, and online-testing datasets:

1. **Download raw PDFs**
2. **Extract text**
3. **Clean metadata**
4. **Chunk text & load into Postgres**
5. **Generate embeddings & eval data**
6. **Simulate online data flow (prod & staging)**
7. **Inspect data via a Streamlit dashboard**

### Folder Splits

Our data is divided in different folder in YYMM format. We generated a list of all the folders of data available to use and split it for training like this:

* **Folders 1–53** → Training & evaluation
* **Folders 54–56** → Retraining batch 1
* **Folders 57–59** → Retraining batch 2

To split the data run in `/data-pipeline-files/split-data` folder:
```
bash create_folders_lists.sh
```
This created 3 file lists with above configuration.
Stored these manually in the object store

### ETL Pipelines

ETL 1-4 and 6 are a standalone bash script and docker service. Run in sequence. ETL 5 conatins a few .ipynb files that are used to generate evaluation data in a particular format (Include papers that have more than 3 citations, is mutually exclusive with the training data and we have all the papers cited by the each paper also in the dataset. This gave us ~13k papers for evaluation dataset data, also the reason why such big data was required to generate eval data) 

Before start working with the ETL pipelines, we need to create a python based docker image that will be used in all the docker compose files after this and have all the required libraries installed in it. Run in `/data-pipeline-files` folder:
```
docker build -t python-gsutil:3.11 .
```

1. **ETL1: Download raw PDF data**

   * **Input:** A text file listing folder URLs (one per line)
   * **Fetch PDFs:** Uses `gsutil` (or equivalent) to pull all PDFs for each folder into a Docker volume
   * **Per-folder workflow:** For **each** folder in your data split, the script will:

     1. **Extract**

        * Download all PDFs in that folder into the designated Docker volume
        
     2. **Transform**

        * Archive the folder of raw PDFs into a single tarball (e.g. `tar -cvf foldername.tar`)

     3. **Load**

        * Upload the resulting tarball to our object store 
        * Clear the downloaded files in docker volume
   * **Execution** (from `/data-pipeline-files/etl1`):

     * **Training & Eval data**

       ```bash
       bash run-etl1.sh
       ```
     * **Retraining batches**

       ```bash
       bash download-test-data.sh
       ```
2. **ETL2: Convert PDF → Text**
   
   * **Input:** Tar file generated in previous step

   * **Convert PDFs to TXT files:** Uses `pymupdf` to extract plain text and also makes use of multiprocessing to make use of the 16 core vm instance available to us to the fullest
   
   * **Store text files to object store:** For **each** text files folder generated in your train/eval split, the script will:

     1. **Extract**

        * Copy the pdf data tar file into the designated Docker volume
        
     2. **Transform**

        * Uncompress the tar file (e.g. `tar -xvf foldername.tar`)
        * Delete the tar file
        * Convert the pdfs to text and store them in a separate folder
        * Compress the text files folder similarly

     3. **Load**
     
        * Upload the resulting tar file to our object store 
        * Clear the files and folders in docker volume
   * **Execution** (from `/data-pipeline-files/etl1`):

     * **Training & Eval data**

       ```bash
       bash run-etl2.sh
       ```
     * **Retraining batches**

       ```bash
       bash transform-pdf-test-data.sh
       ```
3. **ETL3: Clean metadata**

   * **Input:**

     * Raw arXiv metadata JSON lines file (`arxiv-metadata-oai.json`)
     * List of extracted text filenames (`all_files_list.txt`)

   * **Per-run workflow:**

     1. **Extract**

        * Copy the JSON metadata file and create the filenames list into the ETL3 Docker volume

     2. **Transform**

        * Stream through each JSON line in `arxiv-metadata-oai.json`
        * Parse into Python dict, skip malformed lines
        * Deduplicate on `id`
        * Build `txt_filename` by appending the latest version suffix (from `versions`)
        * Filter to only records where:

          * `txt_filename` exists in `all_files_list.txt`
          * Concatenated title+abstract is non-empty and detected as English
        * Extract and normalize fields:

          * `latest_version`
          * `txt_filename`
          * `created_yymm` (from filename)
          * Drop the original `versions` field
        * Write out cleaned records to a single CSV (`arxiv_cleaned_v1.csv`), updating a progress bar

     3. **Load**

        * Upload `arxiv_cleaned_v1.csv` to your object store
        * Remove intermediate JSON and CSV files from the volume

   * **Execution** (from `/data-pipeline-files/etl3`):

     ```
     bash run-etl3.sh
     ```


4. **ETL4: Chunk text & prepare Postgres**

   * **Input:**

     * Cleaned metadata CSV (`arxiv_cleaned_v1.csv`) in `/mnt/object/meta-data`
     * Text tarballs in `/mnt/object/<folder>` (mounted via `DATA_TO_PROCESS`)

   * **Per-folder workflow:**

     1. **Extract**

        * `extract-data` service copies `arxiv_cleaned_v1.csv` from object store into the shared volume
     2. **Transform & Load**

        * Runs `prepare_postgres_data.py`, which for each tarball:

          * Creates/validates the `arxiv_metadata` and `arxiv_chunks_backup` tables
          * Loads the cleaned CSV into `arxiv_metadata` if the table is empty
          * Extracts and cleans text files from the tarball
          * Splits cleaned text into 650-word chunks
          * **Connects to Postgres** (with `PGVECTOR` extension enabled) and:

            * Creates/updates tables: `papers` (aka `arxiv_metadata`) and `text_chunks` (aka `arxiv_chunks_backup`)
            * Bulk-ingests chunks via `COPY` in batches
            * Runs `VACUUM ANALYZE` on the tables to refresh planner statistics
          * Cleans up all extracted folders and intermediate files

   * **Execution** (from `/data-pipeline-files/etl4`):

     * **Training & Eval data**

       ```bash
       bash run-etl4.sh
       ```
     * **Retraining batches**

       ```bash
       bash process-train-and-retraining-data.sh
       ```

5. **ETL5: Embeddings & eval data**

   * **Input:** `text_chunks` table in Postgres

   * **Per-batch workflow:** Process new or sampled chunks:

     1. **Extract**

        * Query Postgres for target chunk IDs (e.g. eval set)
        * Export rows to local JSON Lines

     2. **Transform**

        * Load your trained model in PyTorch
        * Generate vector embeddings for each text chunk (batch & multiprocessing)
        * Attach ground-truth labels from metadata

     3. **Load**

        * Write embeddings back into Postgres `embeddings` table (via `COPY`)
        * Export final eval dataset (embeddings + labels) as CSV/Parquet to object store
        * Clean up local files

   * **Execution** (from `/data-pipeline-files/etl5`):

     ```bash
     bash run-etl5.sh
     ```

6. **ETL6: Online data testing pipeline**

   * **Input:**

     * `queries.txt` file in your object store under `/mnt/object/testing-queries`
     * Environment flag (`staging` or `production`) passed as `DATA_TO_PROCESS`

   * **Simulation workflow:**

        * `queries.txt` was generated manually and uploaded manually to object store
        * Mount `queries.txt` into the container and copy it into the work dir
        * Parse the `DATA_TO_PROCESS` variable to choose `staging` vs. `production`
        * Construct the target table name (`staging_data_queries_online` or `production_data_queries_online`)
        * Connect to Postgres (with `PGVECTOR` enabled) and `CREATE TABLE IF NOT EXISTS …` to ensure the table exists
        * Read all lines from `queries.txt`
        * Split into two batches: first 50 for staging, next 50 for production
        * For each query, generate a UUID (`query_id`) and call `mock_backend()` to get a summary, top-k papers, and rating


        * Prepare an `INSERT … RETURNING 1` SQL statement
        * In a per-row transaction, insert into the chosen table
        * Log the returned `rowcount` for success/failure
        * Sleep `DELAY_SECONDS` between inserts to simulate real-time ingestion

   * **Execution** (from /data-pipeline-files/etl6):

     * **For staging testing**

       ```bash
       bash start-staging-testing.sh
       ```
     * **For production testing**

       ```bash
       bash start-production-testing.sh
       ```


### Streamlit Dashboard

An interactive Streamlit app to inspect data quality and distribution across your processed datasets.

**What it shows:**

* **High-Level Metrics**
  • Total records & unique submitters
  • Average abstract length
  • Overall percentage of missing values

* **Time Series**
  • Monthly record counts plotted over time

* **Data-Quality Charts**
  • Missing-value percentages by column
  • Duplicate-record rate

* **Top Entities**
  • Top 10 submitters by record count

* **Category Breakdown**
  • Frequency of each subject category

* **Abstract Length Distribution**
  • Histogram with box-plot overlay for abstract lengths

* **Interactive Filtering**
  • Sidebar controls: filter by category and version, instantly updating all metrics and charts

* **Sample Preview**
  • Scrollable table showing the first 100 records (post-filter)

**Special Modes:**

* **JSON lists**
  • Shows counts and previews of list items from JSON files
* **Citation data**
  • Displays citation counts, distribution histograms, and top-cited papers
* **Query lists**
  • Provides word-count metrics and a search interface over your `queries.txt`

**Execution** (from `/data-pipeline-files/dashboard`):

```bash
bash run-dashboard.sh
```

### Getting Started

#### Prerequisites

* Python 3.9–3.12
* PostgreSQL (with `pgvector` extension)
* Docker & Docker Compose
* `gsutil` CLI
* Streamlit
