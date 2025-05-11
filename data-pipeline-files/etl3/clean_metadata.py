# runs in jupyter container 
import os
import numpy as np
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from tqdm import tqdm
import csv

input_path = os.path.join('/data/meta-data/', 'arxiv-metadata-oai.json')
output_path = os.path.join('/data/meta-data', 'arxiv_cleaned_v1.csv')
txt_filenames = os.path.join('/dir_data', 'all_files_list.txt')

print("Input Path:", input_path)
print("Output Path:", output_path)
print("PDF Filenames TXT:", txt_filenames)

def is_english(text):
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

def extract_latest_version_info(versions):
    if not versions or not isinstance(versions, list):
        return "", ""
    
    # Assume latest is the last string in the list
    latest_version = versions[-1]
    
    if isinstance(latest_version, str):
        return latest_version, ""  # No created date info available
    elif isinstance(latest_version, dict):
        return latest_version.get('version', ''), latest_version.get('created', '')
    else:
        return "", ""

txt_filenames_set = set()
with open(txt_filenames, 'r', encoding='utf-8') as f:
    for line in f:
        filename = line.strip()
        if filename:
            txt_filenames_set.add(filename)

print(f"Loaded {len(txt_filenames_set)} PDF filenames.")

with open(input_path, 'r', encoding='utf-8') as f:
    total_lines = sum(1 for _ in f)

print(f"Total lines in metadata file: {total_lines}")

english_count = 0
fieldnames_written = False

with open(input_path, 'r', encoding='utf-8') as f:
    for i in range(1):
        line = f.readline()
        if not line:
            break  # End of file reached before 5 lines
        try:
            record = json.loads(line)
            print(json.dumps(record, indent=2))  # Pretty-print each record
        except json.JSONDecodeError:
            print(f"⚠️ Line {i+1}: Invalid JSON, skipping...")


with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = None
    seen_ids = set()  # Track written IDs

    # Count total lines for progress bar
    with open(input_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    progress = tqdm(infile, total=total_lines, desc="Stored: 0", dynamic_ncols=True)

    for line in progress:
        try:
            record = json.loads(line)
            record_id = record.get('id')

            # Skip if id is missing or already seen
            if not record_id or record_id in seen_ids:
                continue

            text = f"{record.get('title', '')} {record.get('abstract', '')}".strip()

            # Extract version info
            latest_version, latest_created = extract_latest_version_info(record.get("versions", []))
            txt_filename = f"{record_id}{latest_version}.txt"

            # Check filename existence and English language
            if txt_filename in txt_filenames_set and text and is_english(text):
                seen_ids.add(record_id)  # Mark id as seen
                record["latest_version"] = latest_version
                # record["latest_created"] = latest_created  # Uncomment if needed
                record["txt_filename"] = txt_filename
                record["created_yymm"] = txt_filename.split('.')[0]
                record.pop("versions", None)

                if writer is None:
                    fieldnames = list(record.keys())
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()

                writer.writerow(record)
                english_count += 1
                progress.set_description(f"Stored: {english_count}")

        except (json.JSONDecodeError, UnicodeEncodeError, KeyError):
            continue  # Skip corrupted lines or missing keys

print(f"\n English records written (deduplicated): {english_count}")
print(f"📁 Cleaned data saved to: {output_path}")



