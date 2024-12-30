import os

from langchain_cohere import ChatCohere
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


def create_llm(model):

    if model["type"] == "deepseek":
        return ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=os.environ.get("DEEPSEEK_API_KEY"),
            openai_api_base="https://api.deepseek.com",
            max_tokens=8192,
        )
    if model["type"] == "openai":
        return ChatOpenAI(**model["config"])
    if model["type"] == "cohere":
        return ChatCohere(**model["config"])
    if model["type"] == "llama":
        return ChatOllama(**model["config"])
    raise ValueError(f"Invalid model: {model['type']}")


def create_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
