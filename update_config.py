import os
from icecream import ic
import yaml

from config import get_all_variables_in_master_config
from config import REPO_PATH

master_config = get_all_variables_in_master_config()

MASTER_CONFIG = os.path.join(REPO_PATH, "config.yaml")


def update_master_config():
    print("Updating master config")
    with open(MASTER_CONFIG, "w+") as outfile:
        yaml.dump(master_config, outfile)
        outfile.close()
    print("Master config updated")


def update_all_configs():
    update_master_config()


if __name__ == "__main__":
    update_all_configs()