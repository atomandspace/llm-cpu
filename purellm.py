from typing import List
from pathlib import Path
import os
import yaml

from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import chainlit as cl

import warnings

warnings.filterwarnings("ignore")


def get_configs():
    with open(os.path.join(os.path.dirname(__file__), "config.yaml")) as infile:
        config_ = yaml.safe_load(infile)
        infile.close()
    return config_


local_config = get_configs()
model_path = local_config["MODEL_PATH"]

config = {
    'max_new_tokens': 1024,
    'repeatition_penalty': 1.1,
    'temperature': 0,
    'top_k': 50,
    'top_p': 0.9,
    'stream': True,
    'threads': int(os.cpu_count() / 2)
}

llm_init = CTransformers(
    model=model_path,
    model_type='llama',
    lib='avx2',
    **config
)

template = """Question: {question}
Let's think step by step and answer truthfully.
"""


@cl.on_chat_start
def main():
    prompt = PromptTemplate(
        template=template, input_variables=["question"]
    )
    llm_chain = LLMChain(prompt=prompt, llm=llm_init, verbose=True)

    # cache the use session
    cl.user_session.set("llm_chain", llm_chain)


@cl.on_message
async def main(message: str):
    # retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")

    # call the chain asynchronously
    results = await llm_chain.acall(
        message,
        callbacks=[cl.AsyncLangchainCallbackHandler()]
    )

    # return the result
    await cl.Message(content=results["text"]).send()
