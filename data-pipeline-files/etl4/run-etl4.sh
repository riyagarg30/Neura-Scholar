docker compose -f docker-compose-etl4.yaml up extract-data

DATA_TO_PROCESS="text-files-data" \
META_DATA_TABLE="arxiv_metadata" \
CHUNKS_DATA_TABLE="arxiv_chunks" \
docker compose -f docker-compose-etl4.yaml up transform-data

docker rm etl_4_extract_data
docker rm etl_4_transform_data

docker volume rm clean-meta-data_raw-data
