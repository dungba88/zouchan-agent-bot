import logging
import os
import time

from langchain_aws import ChatBedrockConverse
from langchain_cohere import ChatCohere
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


def create_llm(model):
    return CachedLLMDelegate(model)


def _create_llm(model):
    logging.info(f"Creating new model {model}")
    if model["type"] == "deepseek":
        return ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=os.environ.get("DEEPSEEK_API_KEY"),
            openai_api_base="https://api.deepseek.com",
            max_tokens=8192,
        )
    if model["type"] == "bedrock":
        return ChatBedrockConverse(**model["config"])
    if model["type"] == "openai":
        return ChatOpenAI(**model["config"])
    if model["type"] == "cohere":
        return ChatCohere(**model["config"])
    if model["type"] == "llama":
        return ChatOllama(**model["config"])
    raise ValueError(f"Invalid model: {model['type']}")


def create_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


class CachedLLMDelegate:
    """
    A delegate LLM instance that caches the actual LLM for 30 minutes.
    If accessed after the cache expires, it refreshes the LLM instance.
    """

    def __init__(self, model, cache_duration_seconds=1800):
        """
        Args:
            model: The LLM model config
            cache_duration_seconds (int): The cache duration in seconds (default: 1800 seconds = 30 minutes).
        """
        self.model = model
        self.cache_duration_seconds = cache_duration_seconds
        self._cached_llm = None
        self._cache_timestamp = None

    def _get_cached_llm(self):
        """Returns the cached LLM instance, refreshing it if necessary."""
        if (
            self._cached_llm is None
            or self._cache_timestamp is None
            or (time.time() - self._cache_timestamp) > self.cache_duration_seconds
        ):
            logging.info(f"Refreshing LLM model: {self.model}")
            self._cached_llm = _create_llm(self.model)
            self._cache_timestamp = time.time()
        return self._cached_llm

    def __getattr__(self, name):
        """
        Delegate attribute access to the wrapped instance.

        :param name: The name of the attribute.
        :return: The attribute from the wrapped instance.
        """
        return getattr(self._get_cached_llm(), name)
