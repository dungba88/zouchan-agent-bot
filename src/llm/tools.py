from datetime import datetime
from typing import List
from urllib.parse import urlparse, parse_qs

import yt_dlp
from langchain.agents import tool
from langchain_community.agent_toolkits import GmailToolkit
from langchain_community.document_loaders import RSSFeedLoader, WebBaseLoader
from langchain_community.tools import TavilySearchResults
from langchain_core.documents import Document
from langchain_google_community.gmail.create_draft import GmailCreateDraft
from langchain_google_community.gmail.send_message import GmailSendMessage

from langchain_text_splitters import CharacterTextSplitter
import logging

from pydantic import BaseModel, Field
from scholarly import scholarly
from youtube_transcript_api import YouTubeTranscriptApi

from config import (
    BOT_NAME,
    AGENT_LANGUAGE,
    PROMPT_TEMPLATE,
    AGENT_PERSONALITY,
    SUB_LLM_MODEL,
    MAIN_LLM_MODEL,
    TAVILY_ENABLED,
    GMAIL_ENABLED,
    LANGSMITH_ENABLED,
)
from llm.chains.gmail_newsletter import GmailThreadSummarizer
from llm.utils import create_llm
from utils.db import insert_doc, load_documents_from_db, is_doc_exist
from utils.indexing import get_indexer_instance


class SummarizeTextSchema(BaseModel):
    text: str = Field(description="The text to summarize")


class GetChannelVideosSchema(BaseModel):
    channel_url: str = Field(
        description="The URL of the YouTube channel (e.g., 'https://www.youtube.com/@CHANNEL_NAME')"
    )


class ExtractYTTranscriptSchema(BaseModel):
    video_url: str = Field(
        description="The URL of the YouTube video from which to extract the transcript."
    )


class ExtractKeywordsSchema(BaseModel):
    content: str = Field(description="the text to extract the keywords")


class FetchRSSSchema(BaseModel):
    urls: list = Field(description="the list of URLS")


class SearchResearchPapersSchema(BaseModel):
    query: str = Field(description="the query to search for")
    max_results: int = Field(
        description="maximum number of results to return, default to 10", default=10
    )


class LoadWebpageSchema(BaseModel):
    url: str = Field(description="the URL to get the content")


class QueryArticlesSchema(BaseModel):
    query: str = Field(description="The search query string.")
    max_results: int = Field(
        description="The maximum number of documents to return.", default=50
    )
    published_date: str = Field(
        description=" The minimum published date (YYYY-MM-DD) for filtering",
        default=None,
    )


def split_doc(doc):
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.create_documents([doc.page_content], metadatas=[doc.metadata])


@tool(args_schema=SearchResearchPapersSchema)
def search_for_research_paper(query: str, max_results=10):
    """
    Search for research (academic) papers using given query and max_results
    """
    search_query = scholarly.search_pubs(query)

    papers = []
    for _ in range(max_results):
        paper = next(search_query)

        # Retrieve details for each paper
        title = paper.get("bib", {}).get("title", "No Title Available")
        authors = paper.get("bib", {}).get("author", "No Authors Available")
        abstract = paper.get("bib", {}).get("abstract", "No Abstract Available")
        url = paper.get("pub_url", "No URL Available")
        pdf_url = paper.get("eprint_url", "No PDF Available")
        pub_year = paper.get("bib", {}).get("pub_year", "Unknown")

        papers.append(
            {
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "publication_year": pub_year,
                "pdf_url": pdf_url,
                "url": url,
            }
        )

    return papers


@tool(args_schema=SummarizeTextSchema)
def summarize_text(text: str):
    """
    Summarize a given text
    """
    # Generate the summary
    prompt = f"{PROMPT_TEMPLATE}. Summarize the following text in {AGENT_LANGUAGE}:\n\n{text}"
    keywords = create_llm(model=SUB_LLM_MODEL).predict(prompt)
    return keywords.strip()


@tool(args_schema=GetChannelVideosSchema)
def get_channel_video_list(channel_url: str):
    """
    Fetches the list of videos for a YouTube channel
    as a list of dictionaries containing video metadata (title, url, view_count).
    Returns an empty list if no videos are found or if there is an error.
    """
    try:
        # Setup yt-dlp options
        ydl_opts = {
            "quiet": True,  # Suppress unnecessary output
            "extract_flat": True,  # Avoid downloading videos, only fetch metadata
            "force_generic_extractor": True,  # Ensure it works for channel URLs
        }

        # Create a yt-dlp instance and extract info
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract video metadata from the channel URL
            info_dict = ydl.extract_info(channel_url, download=False)

            # Check if 'entries' is in the extracted data (which contains the list of videos)
            if "entries" in info_dict:
                videos = []
                # Return list of videos (each entry contains metadata for a video)
                for playlist in info_dict["entries"]:
                    for video in playlist["entries"]:
                        videos.append(
                            {
                                "title": video["title"],
                                "url": video["url"],
                                "view_count": video["view_count"],
                            }
                        )
                return videos
            else:
                return []  # No videos found or invalid channel URL
    except Exception as e:
        logging.error(f"Error fetching channel videos: {e}")
        return []


@tool(args_schema=ExtractYTTranscriptSchema)
def summarize_yt_transcript(video_url: str) -> str:
    """
    Extracts the transcript content (if available) from a YouTube video URL and summarize it.
    You must not pass the video channel. This is more efficient if you want to access the summary of the video.

    This function supports various YouTube URL formats, including standard YouTube links
    (e.g., 'https://www.youtube.com/watch?v=VIDEO_ID'), mobile YouTube links
    (e.g., 'https://m.youtube.com/watch?v=VIDEO_ID'), and shortened YouTube links
    (e.g., 'https://youtu.be/VIDEO_ID').
    """
    return summarize_text(extract_yt_transcript(video_url))


@tool(args_schema=ExtractYTTranscriptSchema)
def extract_yt_transcript(video_url: str) -> str:
    """
    Extracts the transcript content (if available) from a YouTube video URL.
    You must not pass the video channel.

    This function supports various YouTube URL formats, including standard YouTube links
    (e.g., 'https://www.youtube.com/watch?v=VIDEO_ID'), mobile YouTube links
    (e.g., 'https://m.youtube.com/watch?v=VIDEO_ID'), and shortened YouTube links
    (e.g., 'https://youtu.be/VIDEO_ID').
    """
    valid_domains = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}
    parsed_url = urlparse(video_url)
    domain = parsed_url.netloc.lower()
    if domain not in valid_domains:
        raise RuntimeError(f"Invalid domain {domain}, must be one of {valid_domains}")

    if domain == "youtu.be":
        video_id = parsed_url.path.lstrip("/")
        if not video_id:
            raise RuntimeError(f"Cannot get video_id from url {video_url}")

    # For other domains, extract the 'v' parameter
    query_params = parse_qs(parsed_url.query)
    video_id = query_params.get("v", [None])[0]

    logging.info(f"Extracted video_id: {video_id}")

    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "vi"])

    # Combine the transcript into a single string
    return " ".join([entry["text"] for entry in transcript])


@tool(args_schema=ExtractKeywordsSchema)
def extract_keywords(content: str) -> str:
    """
    Extract keywords from a text and return a comma-separated values of keywords
    """
    prompt = f"Extract important keywords or topics from the following text as comma-separate values:\n\n{content}"
    keywords = create_llm(model=SUB_LLM_MODEL).predict(prompt)
    return keywords.strip()


def enhance_doc(doc, source):
    doc.metadata["source"] = source
    doc.metadata["keywords"] = extract_keywords(doc.page_content)
    return split_doc(doc)


@tool(args_schema=FetchRSSSchema)
def fetch_rss(urls: list) -> int:
    """
    Fetch news RSS from list of URLs, break the article by paragraph and return the number of newly added paragraphs since the last fetch
    Example: fetch_rss(["http://example.com/rss", "http://news.com/rss"])
    """
    counter = 0
    for url in urls:
        logging.info(f"Fetching RSS from {url}")
        rss_loader = RSSFeedLoader(urls=[url])
        documents = rss_loader.load()

        for doc in documents:
            if is_doc_exist(doc):
                continue
            enhanced_docs = enhance_doc(doc, f"rss:{url}")
            for the_doc in enhanced_docs:
                insert_doc(the_doc)
                counter = counter + 1
    return counter


@tool
def reindex():
    """
    Reindex the documents and return the number of indexed documents
    """
    loaded_documents = load_documents_from_db()

    logging.info(f"Reindexing total {len(loaded_documents)} documents...")

    get_indexer_instance().init_with_docs(loaded_documents)

    logging.info(f"{len(loaded_documents)} documents indexed")
    return len(loaded_documents)


@tool
def answer_name():
    """
    Call this when someone asks for your name.
    Return the name.
    """
    return BOT_NAME


@tool
def print_system_config():
    """
    Call this when someone asks for your system configuration.
    """
    return {
        "MODELS": {
            "MAIN_LLM": MAIN_LLM_MODEL,
            "SUB_LLM": SUB_LLM_MODEL,
        },
        "BOT_NAME": BOT_NAME,
        "AGENT_LANGUAGE": AGENT_LANGUAGE,
        "AGENT_PERSONALITY": AGENT_PERSONALITY,
        "PROMPT_TEMPLATE": PROMPT_TEMPLATE,
        "SERVICES_ENABLED": {
            "TAVILY_ENABLED": TAVILY_ENABLED,
            "GMAIL_ENABLED": GMAIL_ENABLED,
            "LANGSMITH_ENABLED": LANGSMITH_ENABLED,
        },
        "TOOLS": {
            "USER_TOOLS": [the_tool.name for the_tool in TOOLS],
            "ADMIN_TOOLS": [the_tool.name for the_tool in ADMIN_TOOLS],
        },
    }


@tool(args_schema=LoadWebpageSchema)
def load_webpage(url: str):
    """
    Summarize a Web page
    """
    documents = WebBaseLoader(url).load()
    return [
        {"metadata": doc.metadata, "page_content": summarize_text(doc.page_content)}
        for doc in documents
    ]


@tool(args_schema=QueryArticlesSchema)
def query_articles_tool(
    query: str, max_results: int = 50, published_date: str = None
) -> List[Document]:
    """
    Find news articles of a given topic with configured max results and date filtering.
    :return: A list of documents matching the criteria.
    """
    # Parse the published_date into a datetime object
    full_date = f"{published_date} 00:00:00"
    min_date = datetime.strptime(full_date, "%Y-%m-%d %H:%M:%S")

    # Step 1: Perform similarity search
    similar_docs = get_indexer_instance().vector_store.similarity_search(
        query, k=max_results
    )

    # Step 2: Filter based on the published_date
    filtered_docs = [
        doc
        for doc in similar_docs
        if "published_date" in doc.metadata
        and datetime.strptime(doc.metadata["published_date"], "%Y-%m-%d %H:%M:%S")
        > min_date
    ]

    return filtered_docs


TOOLS = [
    # summarize_text,
    answer_name,
    search_for_research_paper,
    get_channel_video_list,
    extract_yt_transcript,
    summarize_yt_transcript,
    extract_keywords,
    load_webpage,
    query_articles_tool,
]
if TAVILY_ENABLED:
    tavily_search_tool = TavilySearchResults(
        max_results=10,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False,
        include_images=True,
        # include_domains=[...],
        # exclude_domains=[...],
        # name="...",            # overwrite default tool name
        # description="...",     # overwrite default tool description
        # args_schema=...,       # overwrite default args_schema: BaseModel
    )
    TOOLS.append(tavily_search_tool)

if GMAIL_ENABLED:
    gmail_toolkit = GmailToolkit()
    TOOLS += [
        GmailThreadSummarizer(),
        GmailSendMessage(),
        GmailCreateDraft(),
    ]

ADMIN_TOOLS = [
    fetch_rss,
    reindex,
    print_system_config,
]
