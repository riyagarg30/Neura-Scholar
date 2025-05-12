find /mnt/object/${1}/ -name "*.tar" | \
  xargs -n 1 -P 16 -I{} sh -c 'tar -tf "{}" | grep -v "/$" | xargs -n 1 basename' > all_files_list.txt

docker compose -f docker-compose-etl3.yaml up extract-data
docker compose -f docker-compose-etl3.yaml up transform-data
# docker compose -f docker-compose-etl3.yaml up load-data

DATA_DIRECTORY=${1} \
docker compose -f docker-compose-etl3.yaml up load-data

rm all all_files_list.txt
docker rm etl_3_extract_data
docker etl_3_transform_data
docker etl_3_load_data
