from langchain_core.runnables import Runnable

from llm.chains.articles_newsletter import articles_newsletter
from llm.chains.gmail_newsletter import gmail_newsletter
from llm.chains.refresh_rss import refresh_rss
from llm.chains.weather_notify import weather_notify

CHAINS: dict[str, Runnable] = {
    "gmail_newsletter": gmail_newsletter,
    "refresh_rss": refresh_rss,
    "articles_newsletter": articles_newsletter,
    "weather_notify": weather_notify,
}


def validate_chain(chain):
    if chain not in CHAINS:
        raise ValueError(f"Unrecognized chain {chain}, supported: {CHAINS.keys()}")


def run_chain(chain: str, chain_input: dict):
    validate_chain(chain)
    the_chain = CHAINS[chain]
    return the_chain.invoke(the_chain.input_schema.model_validate(chain_input))
