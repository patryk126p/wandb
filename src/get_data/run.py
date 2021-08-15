"""
Download dataset
"""
import argparse
import logging
import os

import requests
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(job_type="download")
    run.config.update(args)
    logger.info(f"Downloading {args.uri}")
    data = requests.get(args.uri).content.decode("utf-8")
    file_path = os.path.join("data", args.artifact_name)
    with open(file_path, "w") as fh:
        fh.write(data)
    logger.info(f"Uploading {args.artifact_name} to artifact store")
    artifact = wandb.Artifact(
        args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description,
    )
    artifact.add_file(file_path)
    run.log_artifact(artifact)
    artifact.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download URL to a local destination")
    parser.add_argument("uri", type=str, help="URI of file to download")
    parser.add_argument("artifact_name", type=str, help="Name of the downloaded file")
    parser.add_argument("artifact_type", type=str, help="Type of the downloaded file")
    parser.add_argument(
        "artifact_description", type=str, help="Description of the downloaded file"
    )
    args = parser.parse_args()

    go(args)
