#!/bin/bashtch
ls /mnt/object/raw-data/ > tar_files_list.txt

while IFS= read -r line; do
  base=$(basename "$line" .tar)
  echo "basename - $base"  # Output: arxiv_0705

  FILE_URL=${base} \
  DATA_DIRECTORY="text-files-data" \
  PDF_DATA_DIRECTORY="raw-data" \
  docker compose -f docker-compose-etl2.yaml up extract-data

  FILE_URL=${base} \
  DATA_DIRECTORY="text-files-data" \
  PDF_DATA_DIRECTORY="raw-data" \
  docker compose -f docker-compose-etl2.yaml up transform-data

  FILE_URL=${base} \
  DATA_DIRECTORY="text-files-data" \
  PDF_DATA_DIRECTORY="raw-data" \
  docker compose -f docker-compose-etl2.yaml up load-data

done < tar_files_list.txt
