import os
import json
import time
import mlflow
import torch
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses

os.environ["MLFLOW_TRACKING_URI"] = "http://129.114.27.112:8000"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://129.114.27.112:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "admin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin@123"
os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "password"
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("final")

engine = create_engine("postgresql://yugesh:yugesh%40123@129.114.27.112:5432/traindb")
query = text("""
    SELECT query_phrases, chunk_data FROM arxiv_chunks_training_rt
    WHERE query_phrases IS NOT NULL AND LENGTH(TRIM(query_phrases)) > 0
""")
with engine.connect() as conn:
    df = pd.read_sql(query, conn)

train_examples = []
for _, row in df.iterrows():
    try:
        phrases = json.loads(row["query_phrases"]) if isinstance(row["query_phrases"], str) else row["query_phrases"]
        for phrase in phrases:
            if isinstance(phrase, str) and len(phrase.strip()) > 0:
                train_examples.append(InputExample(texts=[phrase.strip(), row["chunk_data"]]))
    except:
        continue

batch_size = 1
epochs = 1

model = mlflow.sentence_transformers.load_model("models:/final/latest")

dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
train_loss = losses.MultipleNegativesRankingLoss(model)

with mlflow.start_run():
    mlflow.log_params({
        "model_name": "1122",
        "train_batch_size": batch_size,
        "epochs": epochs,
        "optimizer": "AdamW",
        "pooling": "mean"
    })

    checkpoint_path = f"./saved_model_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    start = time.time()
    model.fit(
        train_objectives=[(dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=int(0.1 * len(dataloader)),
        show_progress_bar=True,
        use_amp=True,
        optimizer_params={"lr": 2e-5},
        checkpoint_path=checkpoint_path,
        checkpoint_save_steps=len(dataloader)
    )
    mlflow.log_metric("training_time_sec", time.time() - start)

    mlflow.sentence_transformers.log_model(
        model,
        artifact_path="model",
        registered_model_name="final"
    )
