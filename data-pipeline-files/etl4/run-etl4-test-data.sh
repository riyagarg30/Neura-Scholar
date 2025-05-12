#!/bin/bash
docker compose -f docker-compose-etl4.yaml up extract-data

DATA_TO_PROCESS=${1} \
docker compose -f docker-compose-etl4.yaml up transform-data