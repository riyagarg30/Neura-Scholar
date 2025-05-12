#!/bin/bash

data_dir="$1-text"
echo "data dir = $data_dir"
ls /mnt/object/${1}/ > tar_files_list.txt

while IFS= read -r line; do
  base=$(basename "$line" .tar)
  echo "basename - $base"  # Output: arxiv_0705

  FILE_URL=${base} \
  DATA_DIRECTORY=${data_dir} \
  PDF_DATA_DIRECTORY=${1} \
  docker compose -f docker-compose-etl2.yaml up extract-data

  FILE_URL=${base} \
  DATA_DIRECTORY=${data_dir} \
  PDF_DATA_DIRECTORY=${1} \
  docker compose -f docker-compose-etl2.yaml up transform-data

  FILE_URL=${base} \
  DATA_DIRECTORY=${data_dir} \
  PDF_DATA_DIRECTORY=${1} \
  docker compose -f docker-compose-etl2.yaml up load-data

  docker rm etl_transform_data
  docker rm etl_extract_data
  docker rm etl_load_data

done < tar_files_list.txt

rm tar_files_list.txt
