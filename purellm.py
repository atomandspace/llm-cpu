# from langchain.chat_models import ChatOpenAI
import os
import yaml

import chainlit as cl

from langchain.llms import LlamaCpp, CTransformers
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, PromptTemplate

import warnings

warnings.filterwarnings("ignore")


def get_configs():
    with open(os.path.join(os.path.dirname(__file__), "config.yaml")) as infile:
        config = yaml.safe_load(infile)
        infile.close()
    return config


config = get_configs()
model_path = config[["MODEL_PATH"]]

model = LlamaCpp(model_path=model_path, verbose=True)


@cl.on_message
async def main(message: cl.Message):
    # Your custom logic goes here...

    # Send a response back to the user
    await cl.Message(
        content=f"Received: {message.content}",
    ).send()
