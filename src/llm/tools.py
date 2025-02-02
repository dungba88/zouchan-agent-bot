import hashlib
import os
import tempfile
import uuid
from datetime import datetime
from enum import Enum
from math import radians, sin, cos, atan2, sqrt
from typing import List, Type, Optional
from urllib.parse import urlparse, parse_qs, quote, urlencode

import requests
import yt_dlp
from imagekitio import ImageKit
from langchain.agents import tool
from langchain_community.agent_toolkits import GmailToolkit
from langchain_community.document_loaders import (
    RSSFeedLoader,
    PlaywrightURLLoader,
    PyPDFLoader,
)
from langchain_community.tools import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, Tool
from langchain_core.vectorstores import InMemoryVectorStore
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
    AGENT_PERSONALITY,
    SUB_LLM_MODEL,
    MAIN_LLM_MODEL,
    TAVILY_ENABLED,
    GMAIL_ENABLED,
    PLACES_SERVICE,
)
from llm.agents.gmail_newsletter_agent import GmailThreadSummarizer
from llm.utils import create_llm, create_embeddings
from utils.db import insert_doc, load_documents_from_db, is_doc_exist
from utils.indexing import get_indexer_instance


llm = create_llm(model=SUB_LLM_MODEL)


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

    keywords = llm.predict(prompt)
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
    return summarize_text.invoke(
        {
            "text": extract_yt_transcript(video_url),
        }
    )


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
    keywords = llm.predict(prompt)
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
        "AGENT_MAIN_CONFIG": {
            "BOT_NAME": BOT_NAME,
            "AGENT_LANGUAGE": AGENT_LANGUAGE,
            "AGENT_PERSONALITY": AGENT_PERSONALITY,
        },
        "SERVICES_ENABLED": {
            "TAVILY_ENABLED": TAVILY_ENABLED,
            "GMAIL_ENABLED": GMAIL_ENABLED,
            "PLACES_SERVICE": PLACES_SERVICE,
            "LANGSMITH_ENABLED": os.environ.get("LANGCHAIN_TRACING_V2"),
        },
        "AVAILABLE_TOOLS": TOOLS,
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
def load_webpage(url: str, post_content_prompt: str = None):
    """
    Load a Web page and optionally summarize the content if requested.
    """
    if is_pdf(url):
        logging.info("URL is PDF, downloading and loading as PDF instead")
        documents = PDFDownloaderAndLoader(url=url).load()
    else:
        documents = PlaywrightURLLoader(
            urls=[url], headless=False, continue_on_failure=False
        ).load()
    return [
        {
            "metadata": doc.metadata,
            "page_content": summarize_text.invoke(
                {
                    "text": doc.page_content,
                    "additional_summarize_prompt": post_content_prompt,
                }
            ),
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
        "cloud_cover",
        "cloud_cover_low",
        "cloud_cover_mid",
        "cloud_cover_high",
        "weather_code",
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
    min_date = None
    if published_date is not None:
        full_date = f"{published_date} 00:00:00"
        min_date = datetime.strptime(full_date, "%Y-%m-%d %H:%M:%S")

    # Step 1: Perform similarity search
    similar_docs = get_indexer_instance().vector_store.similarity_search(
        query, k=max_results
    )

    if min_date is None:
        return similar_docs

    # Step 2: Filter based on the published_date
    return [
        doc
        for doc in similar_docs
        if "published_date" in doc.metadata
        and datetime.strptime(doc.metadata["published_date"], "%Y-%m-%d %H:%M:%S")
        > min_date
    ]


class FoursquareSearchToolInput(BaseModel):
    query: str = Field(description="Search query, e.g., 'coffee shops'")
    ll: str = Field(
        description="The latitude/longitude around which to retrieve place information. This must be specified as latitude,longitude (e.g., ll=41.8781,-87.6298)."
    )
    radius: int = Field(description="Radius in meters to search", default=5000)
    categories: str = Field(
        description="The Foursquare category ID in comma-separated values to search in",
        default=None,
    )
    max_results: int = Field(description="Maximum results to search in", default=10)


class FoursquareSearchTool(BaseTool):
    name: str = "places_search"
    description: str = (
        "Use this tool to search for places using Foursquare API. "
        "Provide a search query (e.g., 'coffee shops'), a city name and optionally a Foursquare category."
    )
    args_schema: Type[BaseModel] = FoursquareSearchToolInput

    def _run(
        self,
        query: str,
        ll: str,
        radius: int = 5000,
        categories: str = None,
        max_results: int = 10,
    ):
        """
        Executes a search on the Foursquare Places API.
        """
        try:
            url = "https://api.foursquare.com/v3/places/search"
            headers = {"Authorization": f"{self.metadata['api_key']}"}
            params = {
                "query": query,
                "ll": ll,
                "radius": radius,
                "limit": max_results,
                "fields": ",".join(
                    [
                        "fsq_id",
                        "name",
                        "geocodes",
                        "location",
                        "categories",
                        "chains",
                        "related_places",
                        "distance",
                        "closed_bucket",
                        "description",
                        "tel",
                        "website",
                        "hours",
                        "rating",
                        "popularity",
                        "price",
                        "menu",
                        "tastes",
                        "features",
                    ]
                ),
            }
            if categories:
                params["categories"] = categories
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            return [
                {
                    "id": result["fsq_id"],
                    "foursquare_url": f"https://foursquare.com/v/{result['fsq_id']}",
                    "categories": result["categories"],
                    "distance": result["distance"],
                    "geocodes": result["geocodes"],
                    "location": result["location"],
                    "name": result["name"],
                    "related_places": result["related_places"],
                    "is_open": result["closed_bucket"],
                    "description": result.get("description", None),
                    "tel": result.get("tel", None),
                    "website": result.get("website", None),
                    "hours": result.get("hours", None),
                    "rating": result.get("rating", None),
                    "popularity": result.get("popularity", None),
                    "price": result.get("price", None),
                    "menu": result.get("menu", None),
                    "tastes": result.get("tastes", []),
                    "features": result.get("features", []),
                }
                for result in results
            ]

        except requests.exceptions.RequestException as e:
            return f"API request failed: {e}"

    def _arun(self, query: str, city: str):
        """
        Asynchronous version is not implemented.
        """
        raise NotImplementedError("This tool does not support async execution.")


class GooglePlacesSearchToolInput(BaseModel):
    location: str = Field(
        ...,
        description="Location in 'latitude,longitude' format, e.g., '37.7749,-122.4194'.",
    )
    radius: int = Field(2000, description="Search radius in meters, e.g., 1000.")
    query: str = Field(
        None,
        description="Type of places to search, e.g., 'restaurants', 'shopping', 'museums'.",
    )
    open_now: bool = Field(
        False, description="Whether to only show places currently open"
    )
    num_results: int = Field(5, description="Maximum number of results to return")


def calculate_distance(lat1, lng1, lat2, lng2):
    """
    Calculate the distance in meters between two coordinates using the Haversine formula.
    """
    R = 6371000  # Radius of the Earth in meters
    dlat = radians(lat2 - lat1)
    dlng = radians(lng2 - lng1)
    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlng / 2) ** 2
    )
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return round(R * c, 2)


if os.environ.get("IMAGE_KIT_PRIVATE_KEY"):
    IMAGE_KIT = ImageKit(
        public_key=os.environ.get("IMAGE_KIT_PUBLIC_KEY"),
        private_key=os.environ.get("IMAGE_KIT_PRIVATE_KEY"),
        url_endpoint=os.environ.get("IMAGE_KIT_URL"),
    )
else:
    IMAGE_KIT = None


class GooglePlacesSearchTool(BaseTool):
    name: str = "places_search"
    description: str = (
        "A tool to search for places around a specific location using the Google Maps Places API. "
        "Provide the location (latitude and longitude), search radius in meters, and a query such as 'restaurants', 'shopping', or 'museums'."
    )
    args_schema: Type[BaseModel] = GooglePlacesSearchToolInput

    def _run(
        self,
        location: str,
        query: str = None,
        radius: int = 2000,
        open_now: bool = False,
        num_results: int = 5,
    ):
        """
        Executes a search on the Google Places API.
        """
        try:
            url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
            params = {
                "location": location,
                "radius": radius,
                "keyword": query,
                "key": self.metadata["api_key"],
                "opennow": open_now,
            }

            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            user_lat, user_lng = map(float, location.split(","))

            data["results"] = data["results"][:num_results]

            # Parse and format results
            for place in data["results"]:
                lat = place["geometry"]["location"]["lat"]
                lng = place["geometry"]["location"]["lng"]
                place["distance"] = calculate_distance(user_lat, user_lng, lat, lng)
                place["link"] = (
                    f"https://www.google.com/maps/place/?q=place_id:{place['place_id']}"
                )
                if "photos" in place and place["photos"]:
                    place["photos"] = [self.build_photo_url(place["photos"][0])]

            return data["results"]

        except requests.exceptions.RequestException as e:
            return f"API request failed: {e}"

    def build_photo_url(self, photo):
        place_photo_url = "https://maps.googleapis.com/maps/api/place/photo"
        url = f"{place_photo_url}?maxwidth=400&photoreference={photo['photo_reference']}&key={self.metadata['api_key']}"
        if IMAGE_KIT is None:
            return url
        response = IMAGE_KIT.upload_file(
            file=url,
            file_name=f"{hashlib.md5(url.encode()).hexdigest()}.jpg",
        )
        return response.thumbnail_url

    def _arun(self, location: str, radius: int, query: str):
        """
        Asynchronous version is not implemented.
        """
        raise NotImplementedError("This tool does not support async execution.")


class TravelMode(Enum):
    DRIVING = "driving"
    WALKING = "walking"
    BICYCLING = "bicycling"
    TRANSIT = "transit"


class GoogleRouteToolInput(BaseModel):
    origin: str = Field(
        ...,
        description="The start location, in plain text addresses or 'latitude,longitude' format, e.g., '37.7749,-122.4194'.",
    )
    destination: str = Field(
        ...,
        description="The destination, in plain text addresses or 'latitude,longitude' format, e.g., '37.7749,-122.4194'.",
    )
    travel_mode: TravelMode = Field(
        TravelMode.TRANSIT,
        description="The mode of travel, default is transit (public transport). Note that transit mode in Japan is not available.",
    )
    waypoints: List[str] = Field([], description="List of stops along the way")
    arrival_time: Optional[datetime] = Field(
        None, description="Desired arrival time (cannot be used with departure_time)."
    )
    departure_time: Optional[datetime] = Field(
        None, description="Desired departure time (cannot be used with arrival_time)."
    )


class GoogleRouteTool(BaseTool):
    name: str = "route_search"
    description: str = (
        "A tool to get a route between two locations using Google Maps APIs. "
        "It provides a textual description of the route, the travel time, an image showing the route on a map, "
        "and a link to open the directions in Google Maps. You can specify the travel mode: 'driving', 'walking', "
        "'bicycling', or 'transit' (for public transport, such as train or bus)."
        "Note that for transit mode in Japan is not available"
    )
    args_schema: Type[BaseModel] = GoogleRouteToolInput

    def _run(
        self,
        origin: str,
        destination: str,
        travel_mode: TravelMode = TravelMode.TRANSIT,
        waypoints: List[str] = [],
        departure_time: Optional[datetime] = None,
        arrival_time: Optional[datetime] = None,
    ):
        """
        Retrieves a route between two locations using Google Directions API and generates a map image using Google Static Maps API.
        """
        if arrival_time and departure_time:
            raise ValueError(
                "Only one of 'arrival_time' or 'departure_time' can be specified."
            )

        try:
            # Define API endpoints
            directions_url = "https://maps.googleapis.com/maps/api/directions/json"

            # Step 1: Get route details from Directions API
            directions_params = {
                "origin": origin,
                "destination": destination,
                "mode": travel_mode.value,
                "key": self.metadata["api_key"],
            }
            if waypoints:
                directions_params["waypoints"] = "|".join(waypoints)
            if arrival_time:
                directions_params["arrival_time"] = int(arrival_time.timestamp())
            if departure_time:
                directions_params["departure_time"] = int(departure_time.timestamp())
            response = requests.get(directions_url, params=directions_params)
            response.raise_for_status()
            data = response.json()

            if data["status"] != "OK":
                return f"Error fetching directions: {data}"

            # Extract route details
            route = data["routes"][0]
            legs = route["legs"]

            total_duration = sum(leg["duration"]["value"] for leg in legs) // 60

            # Step 2: Generate a map image using Static Maps API
            path = "|".join(
                f"{step['start_location']['lat']},{step['start_location']['lng']}"
                for leg in legs
                for step in leg["steps"]
            )
            static_map_image_url = self._create_google_maps_image(path)
            google_maps_link = self._create_google_maps_link(
                origin,
                destination,
                waypoints,
                travel_mode,
                arrival_time,
                departure_time,
            )

            # Compile response
            return {
                "route_details": {
                    "total_travel_time": total_duration,
                    "route": route,
                },
                "map_image": static_map_image_url,
                "google_maps_link": google_maps_link,
            }

        except requests.exceptions.RequestException as e:
            return f"API request failed: {e}"

    def _create_google_maps_link(
        self,
        origin: str,
        destination: str,
        stops: Optional[List[str]],
        travel_mode: TravelMode,
        arrival_time: Optional[datetime],
        departure_time: Optional[datetime],
    ) -> str:
        """Create a clickable Google Maps link."""
        base_url = "https://www.google.com/maps/dir/?"
        query_params = {
            "api": 1,
            "origin": origin,
            "destination": destination,
            "travelmode": travel_mode.value,
        }

        # Include waypoints if provided
        if stops:
            waypoints = "|".join([quote(stop) for stop in stops])
            query_params["waypoints"] = waypoints

        # Include time if provided
        if arrival_time:
            query_params["arrival_time"] = int(arrival_time.timestamp())
        elif departure_time:
            query_params["departure_time"] = int(departure_time.timestamp())

        return f"{base_url}{urlencode(query_params)}"

    def _create_google_maps_image(self, path: str):
        static_map_url = "https://maps.googleapis.com/maps/api/staticmap"
        static_map_params = {
            "size": "600x400",  # Adjust as needed
            "path": f"color:blue|weight:5|{path}",
            "key": self.metadata["api_key"],
        }
        return f"{static_map_url}?{requests.compat.urlencode(static_map_params)}"

    def _arun(self, origin: str, destination: str):
        """
        Asynchronous version is not implemented.
        """
        raise NotImplementedError("This tool does not support async execution.")


def wrap_tavily_search(tavily_search_tool: TavilySearchResults):

    def tavily_search(query):
        return tavily_search_tool.api_wrapper.raw_results(
            query,
            tavily_search_tool.max_results,
            tavily_search_tool.search_depth,
            tavily_search_tool.include_domains,
            tavily_search_tool.exclude_domains,
            tavily_search_tool.include_answer,
            tavily_search_tool.include_raw_content,
            tavily_search_tool.include_images,
        )

    return Tool(
        name=tavily_search_tool.name,
        description=tavily_search_tool.description,
        args_schema=tavily_search_tool.args_schema,
        func=tavily_search,
    )


recall_vector_store = InMemoryVectorStore(create_embeddings())


def get_user_id(config: RunnableConfig) -> str:
    # TODO: thread_id is currently passed from requests so there will be potential
    # impersonating attack. we should properly authenticate users
    thread_id = config["configurable"].get("thread_id")
    if thread_id is None:
        raise ValueError("thread_id needs to be provided to save a memory.")

    return thread_id.split("/")[0]


@tool
def save_recall_memory(memories: List[str], config: RunnableConfig) -> List[str]:
    """Save memory about user preferences, context to vectorstore for later semantic retrieval."""
    user_id = get_user_id(config)
    documents = []
    for memory in memories:
        document = Document(
            page_content=memory, id=str(uuid.uuid4()), metadata={"user_id": user_id}
        )
        documents.append(document)
    recall_vector_store.add_documents(documents)
    return memories


@tool
def search_recall_memories(query: str, config: RunnableConfig) -> List[str]:
    """Search for relevant memories about user preferences, contexts."""
    user_id = get_user_id(config)

    def _filter_function(doc: Document) -> bool:
        return doc.metadata.get("user_id") == user_id

    documents = recall_vector_store.similarity_search(
        query, k=3, filter=_filter_function
    )
    return [document.page_content for document in documents]


def initialize_tools():
    tools = [
        # summarize_text,
        search_for_research_paper,
        get_channel_video_list,
        extract_yt_transcript,
        summarize_yt_transcript,
        extract_keywords,
        load_webpage,
        query_articles_tool,
        get_weather,
        fetch_rss,
        reindex,
        print_system_config,
        save_recall_memory,
        search_recall_memories,
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
        tools.append(wrap_tavily_search(tavily_search_tool))

    if GMAIL_ENABLED:
        gmail_toolkit = GmailToolkit()
        tools += [
            GmailThreadSummarizer(),
            GmailSendMessage(api_resource=gmail_toolkit.api_resource),
            GmailCreateDraft(api_resource=gmail_toolkit.api_resource),
        ]

    if PLACES_SERVICE == "foursquare":
        tools.append(
            FoursquareSearchTool(
                metadata={
                    "api_key": os.environ.get("FOURSQUARE_API_KEY"),
                }
            )
        )
    elif PLACES_SERVICE == "google":
        tools.append(
            GooglePlacesSearchTool(
                metadata={
                    "api_key": os.environ.get("GOOGLE_CLOUD_API_KEY"),
                }
            )
        )
        tools.append(
            GoogleRouteTool(
                metadata={
                    "api_key": os.environ.get("GOOGLE_CLOUD_API_KEY"),
                    "navitime_api_key": os.environ.get("NAVITIME_API_KEY"),
                }
            )
        )
    return tools


TOOLS = initialize_tools()

TOOL_MAPPINGS = {tool.name: tool for tool in TOOLS}
