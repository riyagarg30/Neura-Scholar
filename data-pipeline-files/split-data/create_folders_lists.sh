gsutil ls gs://arxiv-dataset/arxiv/arxiv/pdf/ > all_folders_list.txt

python3 split_folder.py

# # Training and offline eval, use first 53 folder of the list

# gsutil ls gs://arxiv-dataset/arxiv/arxiv/pdf/ | head -n 53 > file_list.txt

# # For re-training, use next 3

# # For online training:
# # 1. Staging, next 3
# 2. Production, get the data for given month and year