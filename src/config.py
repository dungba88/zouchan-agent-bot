import logging
import os

from dotenv import load_dotenv


load_dotenv()
logging.info("Environment variables loaded")


# Define the Agent language
AGENT_LANGUAGE = "English"

# Define the bot name
BOT_NAME = "Zou-chan"

# Additional prompt request
AGENT_PERSONALITY = "helpful and caring assistant"

# Whether to enable short-term memory
USE_SHORT_TERM_MEMORY = True

TAVILY_ENABLED = os.environ.get("TAVILY_API_KEY") is not None
PLACES_SERVICE = "google"  # can be foursquare or google
GMAIL_ENABLED = False

LINE_AGENT = "travel_assistant_agent"

# Main LLM model used for decision-making tasks
MAIN_LLM_MODEL = {
    "type": "openai",  # can be bedrock, deepseek, openai, cohere, llama
    "config": {
        "model": "gpt-4o-mini",
        "temperature": 0.5,
    },
}

# Sub LLM model used for small, direct tasks
SUB_LLM_MODEL = {
    "type": "openai",  # can be bedrock, deepseek, openai, cohere, llama
    "config": {
        "model": "gpt-4o-mini",
        "temperature": 0,
    },
}

# Define the SQLite database path
DATABASE_PATH = "../data/documents.db"

# Define the vector index path
INDEX_PATH = "../data/vectorstore/documents"

# Define the static resources path
STATIC_RESOURCES_PATH = "../resources"

# Define the cron directory
CRON_PATH = "../cron"
