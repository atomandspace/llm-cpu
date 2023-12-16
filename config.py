"""Module to handle all the constants and variables throughout the repo"""

import os
import sys
import platform

import yaml
from icecream import ic

import GPUtil
from psutil import virtual_memory, cpu_count, users

"""
Execute the script, if this the first time you are using this repo on the current PC.
"""

# -----------------------------------------------------------------------------------------------
# FIXED PATHS, DON'T CHANGE
REPO_PATH = os.path.abspath(os.path.dirname(_file_))
META_PATH = os.path.join(REPO_PATH, "meta")

# ???????????????????????????????????????????????????????????????????????????????????????????????
# CONFIGURABLE PATHS, CHANGE AS PER YOUR LOCAL SYSTEM
# change these paths according to your local setup, don't commit the changes to git.
# this is the downloaded gguf file that will be used for the RAG and inference
MODEL_PATH = os.path.join(META_PATH, "model", "llama-2-7b-chat.Q5_K_M.gguf")
# this is folder where you can keep the documents on which you want to run RAG.
# create sub-folders if required, or you can directly define the path in your script
DOCS_DIR = os.path.join(META_PATH, "docs", "regulation")
# this is where the converted tokens will be stored. It's recommended not to change this.
VECTORDB_PERSIST_DIR = os.path.join(META_PATH, "ckpt", "vectordb")

# -----------------------------------------------------------------------------------------------
# MODEL CONFIGURATIONS
BINDING_PORT = "8889"
DOCKER_CONTAINER_NAME = "gpt_vector_store"
COLLECTION_NAME = os.path.split(DOCS_DIR)[-1]  # collection name assumes you have named the DOCS_DIR


# based on the type of documents you have used


# -----------------------------------------------------------------------------------------------
# CURRENT SYSTEM PROPERTIES
def get_platform():
    if "DATABRICKS_RUNTIME_VERSION" in sys.path:
        return "adb"
    else:
        return sys.platform  # linux or win32. udp is a linux platform as well


PLATFORM = get_platform()
system = platform.uname()
PC_MODEL = " ".join([
    system.system, system.release, system.machine,
    system.processor, system.version, ])

# CPU PROPERTIES
SYS_USER_VCN = users()[0].name
SYS_MEMORY_GB = int(virtual_memory().available / (1024 ** 3))
SYS_MAX_THREADS = cpu_count()
SYS_MAX_CORES = SYS_MAX_THREADS / 2
SYS_THREADS = int(SYS_MAX_THREADS * 0.75)
SYS_CORES = int(SYS_MAX_CORES * 0.75)

# GPU PROPERTIES
gpus = GPUtil.getGPUs()
NUM_GPU = len(gpus)
GPU_ID = gpus[0].id
GPU_NAME = gpus[0].name
GPU_DRIVER = gpus[0].driver
GPU_TOTAL_MEMORY = gpus[0].memoryTotal
GPU_FREE_MEMORY = gpus[0].memoryFree


def get_all_variables_in_master_config():
    local_vars = locals()
    global_vars = globals()
    all_vars = {**local_vars, **global_vars}
    return {var: all_vars[var] for var in all_vars if not var.startswith("__") and not callable(all_vars[var]) and var.isupper()}


if _name_ == "_main_":
    # create meta dirs, these contain huge files and can't be pushed to repo
    if not os.path.exists(os.path.join(REPO_PATH, "../meta", "ckpt")):
        os.makedirs(os.path.join(REPO_PATH, "../meta", "ckpt"))
    if not os.path.exists(os.path.join(REPO_PATH, "../meta", "ckpt", "vectordb")):
        os.makedirs(os.path.join(REPO_PATH, "../meta", "ckpt", "vectordb"))
    if not os.path.exists(os.path.join(REPO_PATH, "../meta", "model")):
        os.makedirs(os.path.join(REPO_PATH, "../meta", "model"))

    ic(
        gpus,
        NUM_GPU,
        GPU_ID,
        GPU_NAME,
        GPU_DRIVER,
        GPU_TOTAL_MEMORY,
        GPU_FREE_MEMORY,
    )

    ic(
        SYS_USER_VCN,
        SYS_MEMORY_GB,
        SYS_MAX_THREADS,
        SYS_MAX_CORES,
        SYS_THREADS,
        SYS_CORES,
    )

    ic(
        PLATFORM,
        PC_MODEL
    )