import logging
import os
from datetime import datetime, timedelta

from langchain_aws import ChatBedrockConverse
from langchain_cohere import ChatCohere
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


def create_llm(model):
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


def get_week_days(today=datetime.today()):
    start_of_week = today - timedelta(days=today.weekday())  # Monday
    end_of_week = start_of_week + timedelta(days=6)  # Sunday
    return {
        "first_day": str(start_of_week.date()),
        "last_day": str(end_of_week.date()),
    }


def get_month_days(today=datetime.today()):
    first_day = today.replace(day=1)  # First day of the month
    # Calculate the last day of the month
    next_month = today.replace(day=28) + timedelta(
        days=4
    )  # Guaranteed to be in the next month
    last_day = next_month.replace(day=1) - timedelta(
        days=1
    )  # Last day of the current month
    return {
        "first_day": str(first_day.date()),
        "last_day": str(last_day.date()),
    }


def get_year_days(today=datetime.today()):
    first_day = today.replace(month=1, day=1)  # January 1st
    last_day = today.replace(month=12, day=31)  # December 31st
    return {
        "first_day": str(first_day.date()),
        "last_day": str(last_day.date()),
    }


def get_last_monday(today=datetime.now()):
    # Calculate the number of days since the last Monday
    days_since_monday = today.weekday()  # Monday is 0, Sunday is 6
    last_monday = today - timedelta(days=days_since_monday + 7)
    return last_monday.date()
