#!/bin/bash

while IFS= read -r line; do
  echo "file name - ${line}"

  FILE_URL=${line} \
  DATA_DIRECTORY = "raw-data" \
  docker compose -f docker-compose-etl1.yaml up extract-data

  FILE_URL=${line} \
  DATA_DIRECTORY = "raw-data" \
  docker compose -f docker-compose-etl1.yaml up transform-data

  FILE_URL=${line} \
  DATA_DIRECTORY = "raw-data" \
  docker compose -f docker-compose-etl1.yaml up load-data

  #docker compose run --rm -e FILE_URL=$line extract-data #| echo "done"   # 2>&1 | tee "logs/$(basename "$line").log"
done < file_list.txt

docker compose -f docker-compose-etl1.yaml rm -fs etl_extract_data etl_transform_data etl_load_data || true
docker volume rm download-raw-data_raw-data