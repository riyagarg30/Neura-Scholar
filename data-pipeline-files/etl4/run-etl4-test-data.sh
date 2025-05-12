#!/bin/bash
docker compose -f docker-compose-etl4.yaml up extract-data

DATA_TO_PROCESS=${1} \
META_DATA_TABLE=${2} \
CHUNKS_DATA_TABLE=${3} \
docker compose -f docker-compose-etl4.yaml up transform-data

docker rm etl_4_extract_data
docker etl_4_transform_data

docker volume rm clean-meta-data_raw-data
