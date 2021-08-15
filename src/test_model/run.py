"""
Test model on test data
"""
import argparse
import logging

import joblib
import pandas as pd
import wandb
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(job_type="test")
    run.config.update(args)

    logger.info(f"Reading test data: {args.test_data}")
    local_path = wandb.use_artifact(args.test_data).file()
    x = pd.read_csv(local_path)
    y = x.pop(args.target)

    logger.info(f"Loading model from {args.model}")
    model_path = wandb.use_artifact(args.model).file()
    model = joblib.load(model_path)

    logger.info("Evaluating model")
    preds = model.predict(x)
    acc = accuracy_score(y, preds)
    logger.info(f"Accuracy on test data: {acc:.2f}")
    run.summary["accuracy"] = acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test model against test dataset")
    parser.add_argument(
        "test_data",
        type=str,
        help="Name of test dataset artifact",
    )
    parser.add_argument(
        "target",
        type=str,
        help="Name of target variable",
    )
    parser.add_argument(
        "model",
        type=str,
        help="Name of model artifact",
    )
    args = parser.parse_args()

    go(args)
