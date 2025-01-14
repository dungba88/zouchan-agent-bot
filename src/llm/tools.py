import os
import tempfile
from datetime import datetime
from typing import List
from urllib.parse import urlparse, parse_qs

import requests
import yt_dlp
from langchain.agents import tool
from langchain_community.agent_toolkits import GmailToolkit
from langchain_community.document_loaders import (
    RSSFeedLoader,
    PlaywrightURLLoader, PyPDFLoader,
)
from langchain_community.tools import TavilySearchResults
from langchain_core.documents import Document
from langchain_google_community.gmail.create_draft import GmailCreateDraft
from langchain_google_community.gmail.send_message import GmailSendMessage

from langchain_text_splitters import CharacterTextSplitter
import logging

from playwright.sync_api import sync_playwright
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
)
from llm.agents.gmail_newsletter_agent import GmailThreadSummarizer
from llm.utils import create_llm
from utils.db import insert_doc, load_documents_from_db, is_doc_exist
from utils.indexing import get_indexer_instance


class SummarizeTextSchema(BaseModel):
    text: str = Field(description="The text to summarize")
    additional_summarize_prompt: str = Field(
        description="Additional prompt/special instruction for summarization",
        default=None,
    )


class GetChannelVideosSchema(BaseModel):
    channel_url: str = Field(
        description="The URL of the YouTube channel (e.g., 'https://www.youtube.com/@CHANNEL_NAME')"
    )


class GetWeatherSchema(BaseModel):
    lat: float = Field(description="The latitude of the place to get weather data")
    lon: float = Field(description="The longitude of the place to get weather data")


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
    post_content_prompt: str = Field(
        description="Prompt/special instruction for post-processing the webpage content after it has been loaded"
    )


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
def summarize_text(text: str, additional_summarize_prompt: str = None):
    """
    Summarize a given text
    """
    # Generate the summary
    if not additional_summarize_prompt:
        additional_summarize_prompt = "No special instruction"

    prompt = f"""Summarize the text in {AGENT_LANGUAGE}.
        Additional instruction for summarization: {additional_summarize_prompt}
        
        Text to summarize:
        {text}"""

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
    return summarize_text.invoke({
        "text": extract_yt_transcript(video_url),
    })


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
            "LANGSMITH_ENABLED": os.environ.get("LANGCHAIN_TRACING_V2"),
        },
    }


def is_pdf(url: str):
    if url.endswith(".pdf"):
        return True
    try:
        # Send a HEAD request to check the Content-Type
        response = requests.head(url, allow_redirects=True, timeout=10)

        # Get Content-Type from headers
        content_type = response.headers.get("Content-Type", "")
        print(content_type)
        return "application/pdf" in content_type.lower()
    except requests.RequestException as e:
        print(f"Error fetching URL: {e}")
        return False


class PDFDownloaderAndLoader:

    def __init__(self, url: str):
        self.url = url

    def load(self) -> list[Document]:
        """
        Download a PDF from a given URL to a temporary file using Playwright
        and load it as PDF Document.
        """
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(accept_downloads=True)
            page = context.new_page()

            # Navigate to the URL
            with page.expect_download() as download_info:
                try:
                    page.goto(self.url, wait_until="networkidle")
                except:
                    pass

            download = download_info.value
            logging.info(f"Start downloading the file...")

            # Create a temporary file for the PDF
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                logging.info(f"Will save the file to {temp_file.name}")
                download.save_as(temp_file.name)
                browser.close()
                logging.info("File downloaded")
                loader = PyPDFLoader(temp_file.name)

            return loader.load()


@tool(args_schema=LoadWebpageSchema)
def load_webpage(
    url: str, post_content_prompt: str = None
):
    """
    Load a Web page and optionally summarize the content if requested.
    """
    if is_pdf(url):
        logging.info("URL is PDF, downloading and loading as PDF instead")
        documents = PDFDownloaderAndLoader(url=url).load()
    else:
        documents = PlaywrightURLLoader(urls=[url], headless=False, continue_on_failure=False).load()
    return [
        {
            "metadata": doc.metadata,
            "page_content": summarize_text.invoke({
                "text": doc.page_content,
                "additional_summarize_prompt": post_content_prompt,
            }),
        }
        for doc in documents
    ]


@tool(args_schema=GetWeatherSchema)
def get_weather(lat: float, lon: float):
    """Fetch 7-days weather data using Open-Meteo API, starting from today 1AM"""
    weather_url = "https://api.open-meteo.com/v1/forecast"
    hourly_params = [
        "temperature_2m",
        "precipitation",
        "relative_humidity_2m",
        "snowfall",
        "rain",
        "showers",
        "apparent_temperature",
        "wind_speed_10m",
    ]
    daily_params = [
        "temperature_2m_max",
        "temperature_2m_min",
        "apparent_temperature_max",
        "apparent_temperature_min",
        "precipitation_sum",
        "rain_sum",
        "showers_sum",
        "snowfall_sum",
        "wind_speed_10m_max",
    ]
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(hourly_params),
        "daily": ",".join(daily_params),
        "timezone": "auto",
    }
    response = requests.get(weather_url, params=params)
    if response.status_code != 200:
        raise Exception(f"Weather API Error: {response.status_code} - {response.text}")
    results = response.json()
    return {
        "timezone": results["timezone"],
        "elevation": results["elevation"],
        "hourly_units": results["hourly_units"],
        "daily_units": results["daily_units"],
        "hourly": {
            time: {metric: results["hourly"][metric][index] for metric in hourly_params}
            for index, time in enumerate(results["hourly"]["time"])
        },
        "daily": {
            time: {metric: results["daily"][metric][index] for metric in daily_params}
            for index, time in enumerate(results["daily"]["time"])
        },
    }


@tool(args_schema=QueryArticlesSchema)
def query_articles_tool(
    query: str, max_results: int = 50, published_date: str = None
) -> List[Document]:
    """
    Find news/articles of a given topic. Only search in the indexed RSS feeds.
    For broader Internet-scoped search, use Tavily tool.
    For research papers specifically, use search_for_research_paper tool.
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
    search_for_research_paper,
    get_channel_video_list,
    extract_yt_transcript,
    summarize_yt_transcript,
    extract_keywords,
    load_webpage,
    query_articles_tool,
    get_weather,
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
        GmailSendMessage(api_resource=gmail_toolkit.api_resource),
        GmailCreateDraft(api_resource=gmail_toolkit.api_resource),
    ]

ADMIN_TOOLS = [
    fetch_rss,
    reindex,
    print_system_config,
]

TOOL_MAPPINGS = {tool.name: tool for tool in TOOLS + ADMIN_TOOLS}
