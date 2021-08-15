import os

import mlflow
import yaml

_steps = [
    "download",
    "clean",
    "split",
    "train",
]


def go(config: dict):
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config["main"]["steps"]
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    if "download" in active_steps:
        _ = mlflow.run(
            os.path.join("src", "get_data"),
            "main",
            parameters={
                "uri": config["download"]["uri"],
                "artifact_name": config["download"]["artifact_name"],
                "artifact_type": config["download"]["artifact_type"],
                "artifact_description": config["download"]["artifact_description"],
            },
        )

    if "clean" in active_steps:
        _ = mlflow.run(
            os.path.join("src", "clean_data"),
            "main",
            parameters={
                "input_artifact": config["clean"]["input_artifact"],
                "output_artifact": config["clean"]["output_artifact"],
                "output_type": config["clean"]["output_type"],
                "output_description": config["clean"]["output_description"],
                "col_names": config["clean"]["col_names"],
            },
        )

    if "split" in active_steps:
        _ = mlflow.run(
            os.path.join("src", "split_data"),
            "main",
            parameters={
                "input_artifact": config["split"]["input_artifact"],
                "test_size": config["split"]["test_size"],
                "random_seed": config["split"]["random_seed"],
                "file_names": config["split"]["file_names"],
            },
        )

    if "train" in active_steps:
        _ = mlflow.run(
            os.path.join("src", "train_model"),
            "main",
            parameters={
                "train_data": config["train"]["train_data"],
                "target": config["train"]["target"],
                "model_config": config["train"]["model_config"],
                "model_name": config["train"]["model_name"],
            },
        )


if __name__ == "__main__":
    with open("config.yaml", "r") as fh:
        conf = yaml.safe_load(fh)

    go(conf)
