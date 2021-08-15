"""
Clean dataset
"""
import argparse
import logging
import os

import pandas as pd

import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    col_names = args.col_names.split(",")
    logger.info(f"Reading raw data: {args.input_artifact}")
    local_path = wandb.use_artifact(args.input_artifact).file()
    df = pd.read_csv(local_path, header=None, names=col_names)

    logger.info(f"Dropping rows with empty values")
    df.dropna(axis=0, how="any", inplace=True)

    file_path = os.path.join("data", args.output_artifact)
    logger.info(f"Saving output file to {file_path}")
    df.to_csv(file_path, index=False)
    logger.info(f"Uploading {args.output_artifact} to artifact store")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(file_path)
    run.log_artifact(artifact)
    artifact.wait()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data clean_data")
    parser.add_argument("input_artifact", type=str, help="Name of the input artifact")
    parser.add_argument("output_artifact", type=str, help="Name of cleaned file")
    parser.add_argument("output_type", type=str, help="Type of the cleaned file")
    parser.add_argument(
        "output_description", type=str, help="Description of the cleaned file"
    )
    parser.add_argument(
        "col_names", type=str, help="Comma separated list of column names"
    )
    args = parser.parse_args()

    go(args)
