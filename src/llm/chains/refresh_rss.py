from typing import List

from langchain_core.runnables import chain
from pydantic import BaseModel, Field

from llm.tools import fetch_rss, reindex


class RefreshRSSSchema(BaseModel):
    urls: List[str] = Field(description="List of RSS URLs to fetch")


@chain
def refresh_rss(chain_input: RefreshRSSSchema):
    urls = [{"urls": [url]} for url in chain_input.urls]
    fetch_response = fetch_rss.map().invoke(urls)
    reindex_response = reindex.invoke({})
    return {
        "fetch_response": fetch_response,
        "reindex_response": reindex_response,
    }
