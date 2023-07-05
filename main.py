import torch
import yaml
import argparse
import logging.config

from src.hyperparameter.hyperparameter_search import hyperparameter_search
from src.training.train import train
from src.training.test import test

# setup loggers
logging.config.fileConfig("configs/logging.conf", disable_existing_loggers=False)
# get root logger
logger = logging.getLogger(__name__)

# parse arguments
parser = argparse.ArgumentParser(description="RL Challenge")
parser.add_argument(
    "--config", type=str, default="configs/config.yaml", help="config file"
)
args = parser.parse_args()

if __name__ == "__main__":
    # Set pytorch num threads to 1 for faster training.
    torch.set_num_threads(1)

    # Load config file
    config = yaml.load(open(args.config, "r"), Loader=yaml.SafeLoader)

    # check if hyperparameter tuning is enabled
    if (
        "hyperparameter" in config
        and config["hyperparameter"]["do_hyperparameter_search"]
    ):
        logger.info("Start hyperparameter search...")
        hyperparameter_search(config)
    elif "test" in config and config["test"]:
        logger.info("Start testing...")
        test(config)
    else:
        logger.info("Start training...")
        train(config)
