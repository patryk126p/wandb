"""
Train decision tree model
"""
import argparse
import json
import logging
import os

import joblib
import pandas as pd
import wandb
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(job_type="train")
    run.config.update(args)

    logger.info(f"Reading train data: {args.train_data}")
    local_path = wandb.use_artifact(args.train_data).file()
    x = pd.read_csv(local_path)
    y = x.pop(args.target)

    logger.info(f"Reading model config from {args.model_config}")
    with open(args.model_config) as fp:
        model_config = json.load(fp)

    logger.info("Training model")
    run.config.update(model_config)
    model = DecisionTreeClassifier(**model_config)
    model.fit(x, y)

    model_path = os.path.join("model", args.model_name)
    logger.info(f"Saving model to {model_path}")
    joblib.dump(model, model_path)
    logger.info(f"Uploading {args.model_name} to artifact store")
    artifact = wandb.Artifact(
        args.model_name,
        type="model",
        description="decision tree model",
    )
    artifact.add_file(model_path)
    run.log_artifact(artifact)
    artifact.wait()

    logger.info("Calculating accuracy")
    acc = accuracy_score(y, model.predict(x))
    logger.info(f"Accuracy on train data: {acc:.2f}")
    run.summary["accuracy"] = acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train decision tree classifier")
    parser.add_argument("train_data", type=str, help="Train dataset artifact")
    parser.add_argument(
        "target",
        type=str,
        help="Name of target variable",
    )
    parser.add_argument(
        "model_config",
        type=str,
        help="Path to json with model configuration",
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="Name of model artifact",
    )
    args = parser.parse_args()

    go(args)
