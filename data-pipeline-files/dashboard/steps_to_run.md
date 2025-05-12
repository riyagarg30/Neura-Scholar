bash run-dashboard.py

ssh -i ~/.ssh/id_rsa_chameleon \
    -f -N \
    -L 8501:127.0.0.1:8501 \
    cc@129.114.27.68

board on: http://127.0.0.1:8501/
