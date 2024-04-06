from langchain_openai import OpenAI, ChatOpenAI
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache
from langchain_community.llms.ollama import Ollama
from langchain_community.llms.huggingface_hub import HuggingFaceHub
import os

HUGGING_FACE_API_KEY = os.getenv("HUGGINGFACE_TOKEN")


MODEL_MAP = {
    "gpt-3.5-turbo": ChatOpenAI,
    "gpt-4": ChatOpenAI,
    "gpt-4-1106-preview": ChatOpenAI,
    "gpt-3.5-turbo-instruct": OpenAI,
    "llama2": Ollama,
    "mixtral:8x7b": Ollama,
    "mixtral:7b": Ollama,
    "mistral:7b": Ollama,
    "command-r": Ollama,
    "phi": Ollama
}


def get_model(model_name: str, temperature: float, cache: bool, **kwargs) -> OpenAI:
    model_function = MODEL_MAP[model_name]

    if model_function is None:
        raise ValueError(f"Model {model_name} not found in model map")
    if cache:
        set_llm_cache(SQLiteCache(database_path=".langchain.db"))

    if model_name in ["llama2", "mixtral:8x7b", "mistral:7b", "command-r", "phi"]:
        return model_function(model=model_name)
    
    if model_name in ["HuggingFaceH4/zephyr-7b-beta","NousResearch/Hermes-2-Pro-Mistral-7B"]:
        model_kwargs = {
            "max_new_tokens": 512,
            "top_k": 30,
            "temperature": 0.8,
            "repetition_penalty": 1.03,
        }

        return model_function(
            repo_id=model_name,
            task="text-generation",
            cache=cache,
            huggingfacehub_api_token=HUGGING_FACE_API_KEY,
            model_kwargs=model_kwargs,
        )
    return model_function(
        model=model_name, temperature=temperature, cache=cache, **kwargs
    )
