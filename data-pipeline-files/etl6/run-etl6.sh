DATA_TO_PROCESS=${1} \
docker compose -f docker-compose-etl6.yaml up load-prod-staging-data

# docker compose -f docker-compose-etl6.yaml up load-prod-staging-data
docker rm load_prod_staging_data
docker volume rm production-staging-test-queries_raw-data