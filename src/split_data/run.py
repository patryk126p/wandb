"""
Split dataset into train and test
"""
import argparse
import logging
import os

import pandas as pd
from sklearn.model_selection import train_test_split

import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(job_type="split")
    run.config.update(args)

    logger.info(f"Reading clean data: {args.input_artifact}")
    local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(local_path)

    logger.info("Splitting data")
    train, test = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_seed,
    )

    file_names = args.file_names.split(",")
    for part, name, kind in zip([train, test], file_names, ["train", "test"]):
        file_path = os.path.join("data", name)
        logger.info(f"Saving {file_path}")
        part.to_csv(file_path, index=False)
        logger.info(f"Uploading {name} to artifact store")
        artifact = wandb.Artifact(
            name,
            type=f"{kind}_data",
            description=f"{kind} dataset",
        )
        artifact.add_file(file_path)
        run.log_artifact(artifact)
        artifact.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data to train and test sets")
    parser.add_argument("input_artifact", type=str, help="Name of the input artifact")
    parser.add_argument(
        "test_size",
        type=float,
        help="Size of the test split. Fraction of the dataset, or number of items",
    )
    parser.add_argument(
        "random_seed", type=int, help="Seed for random number generator"
    )
    parser.add_argument(
        "file_names",
        type=str,
        help="Comma separated list of names for train and test datasets",
    )
    args = parser.parse_args()

    go(args)
