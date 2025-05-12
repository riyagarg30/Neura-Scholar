find /mnt/object/text-files-data/ -name "*.tar" | \
  xargs -n 1 -P 16 -I{} sh -c 'tar -tf "{}" | grep -v "/$" | xargs -n 1 basename' > all_files_list.txt

docker compose -f docker-compose-etl3.yaml up extract-data
docker compose -f docker-compose-etl3.yaml up transform-data
# docker compose -f docker-compose-etl3.yaml up load-data
DATA_DIRECTORY="text-files-data" \
docker compose -f docker-compose-etl3.yaml up load-data