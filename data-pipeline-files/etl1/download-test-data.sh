chmod +x run-etl1-test-data.sh
./run-etl1-test-data.sh retrain_list retrain-data
sleep 2
./run-etl1-test-data.sh staging_list staging-data