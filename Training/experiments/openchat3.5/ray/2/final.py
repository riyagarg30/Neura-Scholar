import os
import json
import mlflow
import torch
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
from ray import init

import ray


def train_model():
    os.environ["MLFLOW_TRACKING_URI"] = "http://129.114.27.112:8000"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://129.114.27.112:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = "admin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin@123"
    os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "password"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def train_func():
        import os
        import json
        import time
        import pandas as pd
        from datetime import datetime
        from sqlalchemy import create_engine, text
        import mlflow
        from torch.utils.data import DataLoader
        from sentence_transformers import SentenceTransformer, InputExample, losses, models
        import mlflow.sentence_transformers

        os.environ["MLFLOW_TRACKING_URI"] = "http://129.114.27.112:8000"
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://129.114.27.112:9000"
        os.environ["AWS_ACCESS_KEY_ID"] = "admin"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin@123"
        os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "password"
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        mlflow.set_experiment("final")

        #engine = create_engine("postgresql://rg5073:rg5073pass@129.114.26.124:5432/cleaned_meta_data_db")
        engine = create_engine("postgresql://yugesh:yugesh%40123@129.114.27.112:5432/traindb")

        query = text("""
        SELECT query_phrases, chunk_data FROM arxiv_chunks_training_main
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

        model_name = "allenai/longformer-base-4096"
        max_seq_length = 1500
        batch_size = 2
        epochs = 3

        word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        train_loss = losses.MultipleNegativesRankingLoss(model)

        #model_save_path = os.path.abspath(f"./saved_model_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        model_save_path = f"/mnt/object/saved_model_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        with mlflow.start_run():
            mlflow.log_params({
                "model_name": model_name,
                "max_seq_length": max_seq_length,
                "train_batch_size": batch_size,
                "epochs": epochs,
                "optimizer": "AdamW",
                "pooling": "mean"
            })
            start = time.time()
            model.fit(
                train_objectives=[(dataloader, train_loss)],
                epochs=epochs,
                warmup_steps=int(0.1 * len(dataloader)),
                show_progress_bar=True,
                use_amp=True,
                optimizer_params={"lr": 2e-5},
                checkpoint_path=model_save_path,
                checkpoint_save_steps=len(dataloader)
            )
            mlflow.log_metric("training_time_sec", time.time() - start)
            model.save(model_save_path)
            mlflow.log_artifacts(model_save_path, artifact_path="model")
            mlflow.sentence_transformers.log_model(
                model,
                artifact_path="model",
                registered_model_name="final"
            )

    init(ignore_reinit_error=True)

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        scaling_config=ScalingConfig(
            num_workers=1,
            use_gpu=True
        )
    )

    trainer.fit()


if __name__ == "__main__":
    train_model()
