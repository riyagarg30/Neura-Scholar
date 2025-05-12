docker compose -f docker-compose-etl4.yaml up extract-data

DATA_TO_PROCESS="text-files-data" \
docker compose -f docker-compose-etl4.yaml up transform-data