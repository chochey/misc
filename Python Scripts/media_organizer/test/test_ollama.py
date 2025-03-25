"""
Enhanced Media Organizer with Comprehensive Edge Case Handling
"""
import os
import re
import shutil
import requests
import time
import argparse
import socket
import platform
import subprocess
import ollama
import logging
from functools import lru_cache
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import string
from rapidfuzz import fuzz
import json
from concurrent.futures import ThreadPoolExecutor
import unicodedata
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('media_organizer.log')
    ]
)
logger = logging.getLogger(__name__)

# Paths
MAIN_DIR = r"I:\share-downloads\complete"  # Source directory to scan
MOVIES_DIR = r"I:\Media3\Movies"  # Destination for movie files
TV_DIR = r"I:\Media3\TV"  # Destination for TV show files
COLLECTIONS_DIR = r"I:\Media3\Movies"  # New directory for movie collections

# Global configuration
DRY_RUN = True  # Set to False to actually move files, True for simulation only
ALWAYS_USE_LLM = True  # Default to always using LLM for title cleaning
RECURSIVE_SCAN = True  # Scan subdirectories recursively
MAX_SCAN_DEPTH = 5  # Maximum depth for recursive scanning
HANDLE_COLLECTIONS = True  # Enable handling of movie collections
MAX_RETRY_ATTEMPTS = 3  # Maximum retry attempts for API calls or file operations
FALLBACK_LANGUAGE = 'en'  # Default language for title normalization

OLLAMA_MODEL = 'gemma3:1b'  # Default Ollama model
# Available Ollama models
AVAILABLE_OLLAMA_MODELS = {
    'gemma3:1b': 'Gemma 1B model - fastest option',
    'deepseek-r1:1.5b': 'DeepSeek 1.5B model - more accurate'
}

# API configuration
API_URL = "https://imdb.iamidiotareyoutoo.com/search"

# TMDb API configuration
TMDB_API_KEY = ""  # Add your TMDb API key here if you want to use it
TMDB_ENABLED = False  # Set to True to enable TMDb API fallback

# API search configuration
MIN_FUZZ_RATIO = 0.6  # Minimum match quality (0.0-1.0) for accepting API results
MAX_RETRIES = 3  # Maximum number of API retries
RETRY_DELAY = 5  # Delay between retries in seconds
RETRY_STATUS_CODES = [429, 500, 502, 503, 504]  # Status codes to retry
BACKOFF_FACTOR = 0.3  # Backoff factor for retry delay
API_TIMEOUT = 10  # API timeout in seconds
MAX_SEARCH_QUERIES = 10  # Maximum number of search queries to try
SEARCH_DELAY = 0.5  # Delay between searches to avoid rate limiting (seconds)

# Words to ignore when cleaning titles for API search
API_IGNORE_WORDS = [
    "trailer", "sample", "dummy", "garbage", "1080p", "720p", "480p",
    "WEBRip", "WEB", "Complete", "H264", "H265", "10bit", "x264",
    "BluRay", "HDTV", "AAC", "DDP", "HEVC", "AMZN", "DL", "x265",
    "BluRay", "5.1", "YTS", "uncut", "unrated", "french", "eng",
    "rarbg", "yify", "remastered", "remux", "HDR", "UHD", "DTS", "MultiSub", "iTA",
    "MIRCrew", "AsPiDe", "System Volume Information", "Blu-Ray"
]

# Recognized movie collection identifiers
COLLECTION_IDENTIFIERS = [
    "trilogy", "collection", "anthology", "complete collection",
    "quadrilogy", "saga", "duology", "series", "franchise",
    "all movies"
]

# Known TV shows (to handle ambiguous title detection)
KNOWN_TV_SHOWS = [
    "game of thrones", "breaking bad", "the office", "friends", "stranger things",
    "clarkson's farm", "the grand tour", "top gear", "the mandalorian",
    "ted lasso", "succession", "the last of us", "the bear", "the crown",
    "fargo", "true detective", "westworld", "black mirror", "the witcher",
    "bojack horseman", "parks and recreation", "the good place", "better call saul",
    "the simpsons", "family guy", "south park", "the walking dead",
    "doctor who", "sherlock", "the big bang theory", "the handmaid's tale",
    "24", "prison break", "lost", "house of cards", "the x-files", "twin peaks",
    "the sopranos", "mad men", "grey's anatomy", "law & order", "dexter"
]

# Anime-specific terminology
ANIME_TERMS = [
    "OVA", "OAV", "Special", "Specials", "SP", "ONA", "OAD", "OADs",
    "Movie", "Movies", "Cour", "Part", "Arc", "Season", "Episode",
    "Gekijouban", "Film", "Films", "Complete", "Series", "Batch"
]

# Extended TV show indicators for recognition
TV_INDICATORS = [
    "season", " s0", " s1", " s2", " s3", "episode", "s01d", "s02d",
    "complete", "tv show", "series", "miniseries", "mini-series",
    "collection", "trilogy", "box set", "documentary series",
    "limited series", "web series", "tv special", "tv miniseries",
    "the complete series", "the complete collection", "season pack",
    "all episodes", "all seasons", "complete pack", "episodes", "eps"
]

# Special content folders for TV shows
SPECIAL_CONTENT_FOLDERS = [
    "extras", "features", "special features", "featurettes", "deleted scenes",
    "behind the scenes", "interviews", "bloopers", "gag reel", "outtakes",
    "making of", "trailers", "promos", "soundtrack", "ost", "score",
    "concept art", "artwork", "stills", "images", "photos", "gallery",
    "merchandise", "press kit", "promotional", "additional content",
    "bonus", "bonus disc", "special", "specials", "season 0", "extras", "extra",
    "companion"
]

# Movie version indicators
MOVIE_VERSION_INDICATORS = [
    "director's cut", "extended", "theatrical", "unrated", "rated",
    "alternate", "uncensored", "censored", "international", "domestic",
    "criterion", "collector's edition", "special edition", "restored",
    "remastered", "anniversary edition", "final cut", "redux",
    "expanded", "uncut", "4k remaster", "hdr", "dolby vision",
    "directors cut", "extended cut", "limited edition"
]

# File extensions to process
MEDIA_EXTENSIONS = [
    '.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm',
    '.m4v', '.mpg', '.mpeg', '.mp2', '.m2v', '.m4v', '.3gp',
    '.3g2', '.ogv', '.ts', '.mts', '.m2ts', '.divx', '.xvid'
]

# Database for content information (initially empty, will be populated during use)
CONTENT_DATABASE = {
    "movies": {},
    "tv_shows": {},
    "collections": {},
}

# Database file path for persistence
DB_FILE_PATH = "media_organizer_db.json"


def ensure_ollama_running():
    """Check if Ollama is running and start it if not"""
    logger.info("Checking if Ollama service is running...")

    # Try to connect to Ollama's default port (11434)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    result = sock.connect_ex(('127.0.0.1', 11434))
    sock.close()

    if result == 0:
        logger.info("Ollama service is already running.")
        return True

    logger.info("Ollama service is not running. Attempting to start...")

    try:
        # Platform-specific start commands
        system = platform.system()

        if system == 'Windows':
            # On Windows, start Ollama in a separate process
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            # Get the path to the Ollama executable (assuming it's in PATH)
            ollama_path = "ollama.exe"

            # Start Ollama in the background
            process = subprocess.Popen(
                [ollama_path, "serve"],
                startupinfo=startupinfo,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Wait for Ollama to start
            logger.info("Waiting for Ollama service to start...")
            for _ in range(10):  # Try for 10 seconds
                time.sleep(1)
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('127.0.0.1', 11434))
                sock.close()
                if result == 0:
                    logger.info("Ollama service started successfully!")
                    return True

            logger.warning("Started Ollama but service not responding yet.")
            return False

        elif system == 'Darwin':  # macOS
            # On macOS, Ollama might run as a regular process or app
            subprocess.Popen(["open", "-a", "Ollama"])
            time.sleep(3)  # Give it time to start
            return True

        elif system == 'Linux':
            # On Linux, try to start the Ollama service
            subprocess.Popen(["ollama", "serve"],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
            time.sleep(3)  # Give it time to start
            return True

        else:
            logger.error(f"Unsupported platform: {system}")
            return False

    except Exception as e:
        logger.error(f"Error starting Ollama: {str(e)}")
        logger.error("Please start Ollama manually before running this script.")
        return False


def shutdown_ollama():
    """Shutdown Ollama service to free up resources"""
    logger.info("Shutting down Ollama service...")
    try:
        # Platform-specific shutdown commands
        system = platform.system()

        if system == 'Windows':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            subprocess.run(
                ["ollama", "shutdown"],
                startupinfo=startupinfo,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        else:
            # For Linux and macOS
            subprocess.run(["ollama", "shutdown"],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)

        # Verify shutdown by checking if the port is closed
        time.sleep(1)  # Give it a moment to shut down
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('127.0.0.1', 11434))
        sock.close()

        if result != 0:
            logger.info("Ollama service successfully shut down.")
            return True
        else:
            logger.warning("Warning: Ollama may still be running.")
            return False

    except Exception as e:
        logger.error(f"Error shutting down Ollama: {str(e)}")
        return False


def regex_clean_title(folder_name):
    """Improved regex-based cleaning for when Ollama fails"""
    # First try to extract the title and year if already in correct format
    title_year_match = re.search(r'^(.*?\(\d{4}\))', folder_name)
    if title_year_match:
        potential_title = title_year_match.group(1).strip()
        if len(potential_title) > 5:  # Make sure it's a reasonable title
            return potential_title

    # For titles without year info already
    # Add more technical patterns to remove
    cleaned = re.sub(
        r'\b(?:720p|1080p|2160p|HDTV|BluRay|WEB-DL|WEBRip|HDRip|x264|x265|HEVC|10bit|YIFY|RARBG|DD5\.1|AAC|'
        r'DD\d\.\d|Atmos|DSNP|EAC3|[0-9]+MB|GalaxyRG|TGx)\b',
        '', folder_name, flags=re.IGNORECASE)

    # Remove content within square brackets as these often contain release group tags
    cleaned = re.sub(r'\[.*?\]', '', cleaned)

    # Remove content after hyphens as these often denote release groups
    cleaned = re.sub(r'\-.*$', '', cleaned)

    # Handle season info specially - keep it
    season_match = re.search(r'(.*?)(Season \d+|S\d+)', cleaned, re.IGNORECASE)

    if season_match:
        # Clean the title part
        title_part = season_match.group(1).strip()
        season_part = season_match.group(2).strip()

        # Clean title part more aggressively
        title_part = re.sub(r'[.\-_]', ' ', title_part)
        title_part = re.sub(r'\s+', ' ', title_part).strip()

        # Check if title already has year
        year_match = re.search(r'(.*)\((\d{4})\)', title_part)
        if year_match:
            # Already has year, just clean format
            title = year_match.group(1).strip()
            year = year_match.group(2)
            return f"{title} ({year})"

        # Otherwise add year if we can find it in the original title
        year_anywhere = re.search(r'\b(19\d{2}|20\d{2})\b', folder_name)
        if year_anywhere:
            year = year_anywhere.group(1)
            return f"{title_part} ({year})"

        # If no year found, just use title
        return title_part

    # Not a season, just clean up spacing and special chars
    cleaned = re.sub(r'[.\-_]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # If we still have a year in the original, make sure it's included
    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', folder_name)
    if year_match and '(' + year_match.group(1) + ')' not in cleaned:
        year = year_match.group(1)
        return f"{cleaned} ({year})"

    return cleaned


def clean_folder_name(folder_name):
    """Use Ollama Python API to clean and format media titles"""
    # If folder name is very short, likely not a real media title
    if len(folder_name.strip()) < 5:
        logger.warning(f"Folder name too short: '{folder_name}' - Using as is")
        return folder_name

    # First, try to extract title directly if it's already formatted correctly
    direct_extraction = None
    direct_match = re.search(r'^(.*?\(\d{4}\)).*', folder_name)
    if direct_match:
        direct_extraction = direct_match.group(1).strip()
        if len(direct_extraction) > 5:  # Make sure it's a reasonable title
            logger.info(f"Direct title extraction: {direct_extraction}")
            # If not forcing LLM use, return the direct extraction
            if not ALWAYS_USE_LLM:
                return direct_extraction

    # Simplify the title before sending to the LLM
    simplified_title = folder_name
    if len(simplified_title) > 30:
        # Remove all technical specs
        simplified_title = re.sub(r'(?i)(?:720p|1080p|2160p|bluray|web-dl|hdtv|x264|x265|hevc).*$', '',
                                  simplified_title).strip()
        # Remove content in parentheses with technical terms
        simplified_title = re.sub(r'\([^\)]*(?:1080p|720p|HD|BluRay|WEB)[^\)]*\)', '', simplified_title).strip()
        # Remove brackets, dots and underscores
        simplified_title = re.sub(r'[\[\]._]', ' ', simplified_title).strip()
        # Special case for TV shows - extract just the title part
        if 'season' in simplified_title.lower() or 's0' in simplified_title.lower():
            title_part = re.split(r'(?i)season|s\d+|part|cour|episode|ova', simplified_title)[0].strip()
            if title_part:
                simplified_title = title_part
        logger.info(f"Simplified title for LLM: {simplified_title}")

    # Create a more explicit prompt for the LLM with examples
    prompt = f"""
    I need you to format a media title in the standard 'Title (Year)' format.
    Example inputs and outputs:
    - 'Matrix 1999' -> 'The Matrix (1999)'
    - 'Avatar.2009.BluRay' -> 'Avatar (2009)'
    - 'Star Wars A New Hope 1977' -> 'Star Wars: A New Hope (1977)'
    - 'Battlestar Galactica Complete' -> 'Battlestar Galactica (2003)'
    - 'Kraven The Hunter 2024' -> 'Kraven the Hunter (2024)'
    - 'Dragon Ball Z Season 1-9 Complete' -> 'Dragon Ball Z (1989)'
    - 'Breaking Bad S01-S05 Complete 1080p' -> 'Breaking Bad (2008)'
    - 'Star Wars Trilogy 1977 1980 1983' -> 'Star Wars Trilogy (1977-1983)'
    - 'The.Lord.of.the.Rings.Extended.Collection.1080p' -> 'The Lord of the Rings Collection (2001-2003)'
    - 'Doctor Who Season 0 Specials' -> 'Doctor Who (2005)'
    - 'Sherlock_Series_1-4_Complete' -> 'Sherlock (2010)'
    - 'Friends.The.Complete.Series.S01-S10' -> 'Friends (1994)'
    - 'Jujutsu Kaisen S01 Cour 1 1080p' -> 'Jujutsu Kaisen (2020)'
    - 'The Office US Complete Series' -> 'The Office US (2005)'
    - 'Avengers Infinity War and Endgame (2018-2019)' -> 'Avengers Collection (2018-2019)'
    - 'Rick and Morty Season 01-05' -> 'Rick and Morty (2013)'
    - 'Squid Game (Korean Series) Season 1' -> 'Squid Game (2021)'
    - 'Dune Part 1 & Part 2 (2021-2024)' -> 'Dune Collection (2021-2024)'

    Please format this title: {simplified_title}

    IMPORTANT: Output ONLY the title in 'Title (Year)' format with no explanation, no thinking steps, no quotes, and no additional text. Do not start with "<think>" or similar tokens.
    """
    logger.info(f"Sending to Ollama API: {simplified_title}")

    try:
        # Try several times with backoff in case of temporary errors
        max_attempts = 3
        attempt = 1

        while attempt <= max_attempts:
            try:
                # Use Ollama Python API to get a response with improved prompt
                response = ollama.chat(
                    model=OLLAMA_MODEL,  # Use the global model variable
                    messages=[
                        {'role': 'user', 'content': prompt}
                    ],
                    # Set options for faster response
                    options={
                        'temperature': 0.1,  # Low temperature for more deterministic output
                        'num_predict': 50,  # Limit token generation
                    }
                )

                # Extract the response text
                raw_output = response['message']['content'].strip()
                logger.info(f"Raw LLM output:\n{raw_output}")

                # Process the result - extract just the title and year format
                # Remove any thinking tags or reasoning from the output
                if "<think>" in raw_output:
                    # If we have thinking tags, try to extract just the final conclusion
                    lines = raw_output.split('\n')
                    # Look at the last few lines for the actual answer
                    for line in reversed(lines):
                        title_year_match = re.search(r'(.*?)\s*\((\d{4}(?:-\d{4})?)\)', line)
                        if title_year_match:
                            title = title_year_match.group(1).strip()
                            year = title_year_match.group(2)
                            cleaned_title = f"{title} ({year})"
                            logger.info(f"Extracted from thinking: {cleaned_title}")
                            return cleaned_title

                # Try to extract title and year from the raw output directly
                title_year_match = re.search(r'(.*?)\s*\((\d{4}(?:-\d{4})?)\)', raw_output)
                if title_year_match:
                    title = title_year_match.group(1).strip()
                    year = title_year_match.group(2)
                    cleaned_title = f"{title} ({year})"
                    logger.info(f"Extracted from raw output: {cleaned_title}")
                    return cleaned_title

                # If the format doesn't include a year in parentheses, try to find a year and add it
                year_match = re.search(r'\b(19\d{2}|20\d{2})\b', raw_output)
                if year_match:
                    year = year_match.group(1)
                    # Remove year from the text
                    title = re.sub(r'\b' + year + r'\b', '', raw_output).strip()
                    cleaned_title = f"{title} ({year})"
                    logger.info(f"Constructed from raw output and year: {cleaned_title}")
                    return cleaned_title

                # If no year match, just use the raw output if it's reasonable
                if len(raw_output) > 3 and not raw_output.startswith(('<', '[', '{', '/')):
                    logger.info(f"Using raw output as title without year: {raw_output}")
                    # Try to extract year from original filename
                    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', folder_name)
                    if year_match:
                        year = year_match.group(1)
                        cleaned_title = f"{raw_output} ({year})"
                        logger.info(f"Added year from original filename: {cleaned_title}")
                        return cleaned_title
                    return raw_output

                # If we reach here, the LLM didn't provide a usable response
                break

            except Exception as inner_e:
                logger.warning(f"Ollama API attempt {attempt} failed: {str(inner_e)}")
                if attempt < max_attempts:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                attempt += 1

        # If all attempts failed or no valid response, handle fallbacks
        if direct_extraction:
            logger.info(f"LLM output invalid, using direct extraction: {direct_extraction}")
            return direct_extraction
        else:
            logger.info("LLM output invalid, falling back to regex cleaning")
            return regex_clean_title(folder_name)

    except Exception as e:
        logger.error(f"Error using Ollama API: {str(e)}")
        if direct_extraction:
            logger.info(f"Using direct extraction due to error: {direct_extraction}")
            return direct_extraction
        else:
            logger.info("Falling back to regex cleaning")
            return regex_clean_title(folder_name)


# ---------------------- HTTP Session Setup ---------------------- #
def create_session():
    """Create and configure HTTP session with retry handling."""
    session = requests.Session()
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=RETRY_STATUS_CODES
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session


@lru_cache(maxsize=100)
def search_movie(query, api_url, session, timeout=10):
    """Search for a movie or TV show using the API with caching and improved error handling."""
    params = {'q': query}
    try:
        response = session.get(api_url, params=params, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict) or "description" not in data:
            logger.warning(f"Unexpected API response format for '{query}'")
            return None
        return data
    except Exception as e:
        logger.warning(f"API error for query '{query}': {e}")
        return None


def search_tmdb(title, year=None, session=None, media_type=None):
    """Search TMDb API for more reliable TV/Movie classification with enhanced error handling"""
    if not TMDB_ENABLED or not TMDB_API_KEY:
        return None

    logger.info(f"Searching TMDb for: {title}" + (f" ({year})" if year else ""))

    if session is None:
        session = requests.Session()

    # Use retry pattern for robustness
    for attempt in range(MAX_RETRY_ATTEMPTS):
        try:
            # Determine which endpoint to use
            if media_type == "tv":
                search_url = "https://api.themoviedb.org/3/search/tv"
            elif media_type == "movie":
                search_url = "https://api.themoviedb.org/3/search/movie"
            else:
                search_url = "https://api.themoviedb.org/3/search/multi"

            # Format URL for TMDb search
            params = {
                "api_key": TMDB_API_KEY,
                "query": title,
                "include_adult": "false"
            }
            if year:
                params["year"] = year

            response = session.get(search_url, params=params, timeout=API_TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                if "results" in data and data["results"]:
                    results = []

                    # Process each result
                    for item in data["results"]:
                        if "media_type" in item:
                            media_type = item.get("media_type")
                        elif search_url.endswith("/tv"):
                            media_type = "tv"
                        elif search_url.endswith("/movie"):
                            media_type = "movie"
                        else:
                            # If we don't know the media type, skip this item
                            continue

                        if media_type in ["tv", "movie"]:
                            # Extract title based on media type
                            if media_type == "tv":
                                title = item.get("name", "Unknown")
                            else:
                                title = item.get("title", "Unknown")

                            # Extract year from release_date or first_air_date
                            if media_type == "tv" and "first_air_date" in item and item["first_air_date"]:
                                year = item["first_air_date"][:4]
                            elif media_type == "movie" and "release_date" in item and item["release_date"]:
                                year = item["release_date"][:4]
                            else:
                                year = ""

                            # Create result item
                            result_item = {
                                "title": title,
                                "year": year,
                                "type": media_type,
                                "id": item.get("id", ""),
                                "popularity": item.get("popularity", 0),
                                "overview": item.get("overview", ""),
                                "vote_count": item.get("vote_count", 0),
                                "vote_average": item.get("vote_average", 0)
                            }

                            results.append(result_item)

                    # If we have results, return them
                    if results:
                        logger.info(f"Found {len(results)} matches from TMDb")
                        return {"results": results}
                else:
                    logger.info("No results found from TMDb")
            else:
                logger.warning(f"TMDb API returned {response.status_code}: {response.text}")
                # Check if we should retry (rate limit, temporary server issue)
                if response.status_code in [429, 500, 502, 503, 504] and attempt < MAX_RETRY_ATTEMPTS - 1:
                    retry_after = int(response.headers.get('Retry-After', 1))
                    logger.info(f"Retrying in {retry_after} seconds...")
                    time.sleep(retry_after)
                    continue

        except Exception as e:
            logger.error(f"Error searching TMDb: {str(e)}")
            if attempt < MAX_RETRY_ATTEMPTS - 1:
                time.sleep(2 ** attempt)  # Exponential backoff

    # If we reach here, we couldn't get a valid response after retries
    return None


def clean_title_for_search(title):
    """Clean a title for API search by removing technical specifications and unnecessary words."""
    # Get base name without extension
    base = re.sub(r'[._\-]+', ' ', title)

    # Remove quality indicators, encodings, etc.
    title_clean_patterns = [
        r'\bS\d{1,2}E\d{1,2}\b',  # S01E01 format
        r'\bSeason\s*\d+\b',  # Season X
        r'\bEpisode\s*\d+\b',  # Episode X
        r'\b\d{1,2}x\d{1,2}\b',  # 1x01 format
        r'\b(720p|1080p|2160p|480p|HDR|UHD|4K)\b',  # Resolution
        r'\b(x264|x265|HEVC|H264|H265)\b',  # Encoding
        r'\b(BluRay|BRRip|DVDRip|WEBRip|WEB-DL)\b',  # Source
        r'\b(AAC|AC3|DTS|DD5\.1|DDP5\.1)\b',  # Audio
        r'\b(REMASTERED|REMUX|PROPER|REPACK)\b',  # Version
        r'\b\d{1,2}bit\b',  # Bit depth
        r'(?<!\w)(rarbg|yify|YTS|ettv|EZTV)(?!\w)',  # Release groups
    ]

    # Apply all cleaning patterns
    for pattern in title_clean_patterns:
        base = re.sub(pattern, '', base, flags=re.IGNORECASE)

    # Remove API ignore words
    if API_IGNORE_WORDS:
        ignore_pattern = '|'.join(
            r'\b' + re.escape(word) + r'\b'
            for word in API_IGNORE_WORDS
        )
        base = re.sub(ignore_pattern, '', base, flags=re.IGNORECASE)

    # Final cleanup
    base = re.sub(r'\s+', ' ', base).strip()
    base = base.strip(string.punctuation + ' ')
    base = re.sub(r'\(\s*\)', '', base)
    base = re.sub(r'\[\s*\]', '', base)

    # Strip enclosing quotes if present
    if (base.startswith('"') and base.endswith('"')) or \
            (base.startswith("'") and base.endswith("'")):
        base = base[1:-1]

    return base.strip()


def generate_primary_queries(title, year=None):
    """Generate primary search queries from a title and optional year."""
    title = title.strip()
    queries = []

    # Most specific query - exact title with year
    if year:
        queries.append(f"{title} {year}")

    # Just the title
    queries.append(title)

    # Title with media type hints to improve search
    queries.append(f"{title} movie")
    queries.append(f"{title} tv series")

    if year:
        # Try with movie type + year
        queries.append(f"{title} movie {year}")
        # Try with TV type + year
        queries.append(f"{title} tv series {year}")
        # Try with film type + year
        queries.append(f"{title} film {year}")

    return list(dict.fromkeys([q for q in queries if q.strip()]))


def generate_title_variations(title):
    """Generate sensible variations of the title to improve search quality."""
    variations = []
    title = title.strip().lower()

    # Handle common abbreviations and substitutions
    abbreviation_map = {
        "vs": "versus",
        "vs.": "versus",
        "&": "and",
        "+": "and",
        "@": "at",
    }

    # Create version with expanded abbreviations
    mod_title = title
    for abbr, expanded in abbreviation_map.items():
        if f" {abbr} " in f" {mod_title} ":
            mod_title = mod_title.replace(f" {abbr} ", f" {expanded} ")
            if mod_title != title:
                variations.append(mod_title)

    # Handle franchises - try with and without numbering
    franchise_patterns = [
        (r'(\d+)', ''),  # "Star Wars 7" -> "Star Wars"
        (r'part (\d+)', ''),  # "Part 2" -> ""
        (r'chapter (\d+)', ''),  # "Chapter 2" -> ""
        (r'episode (\d+)', ''),  # "Episode 2" -> ""
    ]

    for pattern, replacement in franchise_patterns:
        mod_title = re.sub(pattern, replacement, title)
        mod_title = re.sub(r'\s+', ' ', mod_title).strip()  # Clean up extra spaces
        if mod_title and mod_title != title and len(mod_title) > 4:  # Ensure it's substantial
            variations.append(mod_title)

    # Handle subtitles - remove after colon or dash
    for separator in [":", " - ", " – "]:
        if separator in title:
            main_title = title.split(separator)[0].strip()
            if main_title and main_title != title and len(main_title) > 4:
                variations.append(main_title)

    # For specific problematic titles, add their common variations
    problem_title_map = {
        "pacific rim uprising": ["pacific rim 2", "pacific rim: uprising"],
        "blade runner": ["blade runner 2049", "blade runner: 2049"],
        "star wars": ["star wars episode", "star wars: episode"],
        "thor": ["thor ragnarok", "thor: ragnarok"],
        "clarkson's farm": ["clarksons farm", "clarkson farm", "clarkson's farm tv series"],
        "the grand tour": ["grand tour", "grand tour tv series"],
        "the boys": ["the boys amazon", "the boys tv show"],
        "the bear": ["the bear tv show", "the bear fx"],
        "house of the dragon": ["game of thrones house of the dragon", "hot house of the dragon"],
        "loki": ["loki marvel", "loki mcu", "loki tv series"],
        "the last of us": ["tlou", "the last of us tv", "the last of us hbo"],
        "severance": ["severance apple tv", "severance tv series"],
        "wednesday": ["wednesday addams", "wednesday netflix"],
        "andor": ["star wars andor", "andor star wars"],
        "mandalorian": ["the mandalorian", "star wars mandalorian"],
    }

    for problem_title, variants in problem_title_map.items():
        if problem_title in title:
            variations.extend(variants)

    return list(dict.fromkeys(variations))


def generate_alternative_queries(title, year=None):
    """Generate alternative search queries for fallback."""
    queries = []
    title = title.strip()

    # Remove articles at beginning
    for article in ["the ", "a ", "an "]:
        if title.lower().startswith(article):
            no_article = title[len(article):].strip()
            if year:
                queries.append(f"{no_article} {year}")
            queries.append(no_article)

    # Try with adjacent years
    if year:
        year_int = int(year)
        # Movies sometimes have different production/release years
        queries.append(f"{title} {year_int - 1}")
        queries.append(f"{title} {year_int + 1}")

    # For longer titles, try first significant words
    words = title.split()
    if len(words) > 2:
        # Try first two words for long titles
        first_two = " ".join(words[:2])
        if year:
            queries.append(f"{first_two} {year}")
        queries.append(first_two)

    return list(dict.fromkeys([q for q in queries if q.strip()]))


def enhanced_match_quality(candidate, query, year=None):
    """
    Enhanced scoring system for determining match quality between a query and a candidate.

    Args:
        candidate: Candidate item from API
        query: Original query string
        year: Optional year

    Returns:
        Score from 0-100 indicating match quality
    """
    title = candidate.get("#TITLE", "")
    candidate_year = str(candidate.get("#YEAR", ""))

    # Normalize for comparison
    query_lower = query.lower().strip()
    title_lower = title.lower().strip()

    # Parse title components
    has_subtitle = False
    main_title = title_lower
    subtitle = ""

    # Check for subtitle separators
    for separator in [":", " - ", "–", "—"]:
        if separator in title_lower:
            parts = title_lower.split(separator, 1)
            main_title = parts[0].strip()
            subtitle = parts[1].strip() if len(parts) > 1 else ""
            has_subtitle = True
            break

    # Check for supplementary content indicators
    supplementary_indicators = [
        "behind the scenes", "making of", "special features", "bonus",
        "documentary", "featurette", "interview", "the art of", "the world of",
        "inside", "underworld", "untold story", "bridge to", "path to", "road to",
        "journey to", "commentary", "deleted scenes", "gag reel"
    ]
    is_supplementary = any(indicator in title_lower for indicator in supplementary_indicators)

    # Break into words for analysis
    query_words = set(query_lower.split())
    title_words = set(title_lower.split())
    main_title_words = set(main_title.split())

    # Initialize score components
    title_match_score = 0  # Title match quality (0-40)
    word_match_score = 0  # Word coverage quality (0-15)
    year_match_score = 0  # Year match quality (0-15)
    popularity_score = 0  # Based on vote count (0-10)
    content_bonus = 0  # Bonus for primary content (0-20)
    penalties = 0  # Various penalties (negative)

    # TITLE MATCH SCORING (0-40)

    # Case 1: Exact full match
    if title_lower == query_lower:
        title_match_score = 40

    # Case 2: Main title exact match
    elif main_title == query_lower:
        title_match_score = 38

    # Case 3: Query matches beginning of title
    elif title_lower.startswith(query_lower):
        # Calculate how much of the title is covered by the query
        coverage = len(query_lower) / len(title_lower)
        title_match_score = min(38, 35 * coverage)

    # Case 4: Title matches beginning of query
    elif query_lower.startswith(main_title):
        coverage = len(main_title) / len(query_lower)
        title_match_score = min(35, 30 * coverage)

    # Case 5: Fuzzy matching for other cases
    else:
        ratio = fuzz.ratio(query_lower, title_lower)
        partial_ratio = fuzz.partial_ratio(query_lower, title_lower)
        token_sort_ratio = fuzz.token_sort_ratio(query_lower, title_lower)

        # Use weighted combination of fuzzy metrics
        fuzzy_score = (ratio * 0.3 + partial_ratio * 0.3 + token_sort_ratio * 0.4) / 100
        title_match_score = min(30, 30 * fuzzy_score)

    # WORD MATCH SCORING (0-15)
    if query_words and main_title_words:
        common_words = query_words.intersection(main_title_words)

        if common_words:
            # Word coverage metrics
            query_coverage = len(common_words) / len(query_words)
            title_coverage = len(common_words) / len(main_title_words)

            # Weighted combination favoring query coverage
            combined_coverage = (0.7 * query_coverage) + (0.3 * title_coverage)
            word_match_score = min(15, 15 * combined_coverage)

    # YEAR MATCH SCORING (0-15)
    if year and candidate_year:
        if year == candidate_year:
            year_match_score = 15  # Exact year match
        elif abs(int(year) - int(candidate_year)) == 1:
            year_match_score = 12  # Off by one year
        elif abs(int(year) - int(candidate_year)) <= 2:
            year_match_score = 8  # Off by two years

    # POPULARITY SCORING (0-10)
    vote_count = candidate.get("#VOTE_COUNT", 0)
    if vote_count:
        if vote_count > 100000:
            popularity_score = 10
        elif vote_count > 50000:
            popularity_score = 8
        elif vote_count > 20000:
            popularity_score = 6
        elif vote_count > 5000:
            popularity_score = 4
        elif vote_count > 1000:
            popularity_score = 2

    # CONTENT RELEVANCE BONUS (0-20)
    # Bonus for being main content vs supplementary
    if not is_supplementary:
        content_bonus += 10

        # Additional bonus for reasonable title length
        if len(title_lower) <= len(query_lower) * 1.5:
            content_bonus += 5

    # PENALTIES (negative)
    # Penalty for supplementary content
    if is_supplementary:
        penalties -= 25

    # Smaller penalty for having a subtitle
    elif has_subtitle:
        subtitle_ratio = len(subtitle) / (len(main_title) + len(subtitle))
        penalties -= min(15, 15 * subtitle_ratio)

    # Penalty for excessive length
    if len(title_lower) > len(query_lower) * 2:
        length_ratio = (len(title_lower) - len(query_lower) * 2) / len(title_lower)
        penalties -= min(15, 15 * length_ratio)

    # Calculate final score
    total_score = title_match_score + word_match_score + year_match_score + popularity_score + content_bonus + penalties

    # Ensure score is between 0-100
    return max(0, min(100, total_score))


def process_search_results(results, title, year):
    """Process, deduplicate, and rank search results."""
    # Remove duplicates by IMDB ID
    seen_ids = set()
    unique_results = []

    for item in results:
        imdb_id = item.get("#IMDB_ID")
        if imdb_id and imdb_id not in seen_ids:
            seen_ids.add(imdb_id)
            unique_results.append(item)

    # Score and rank results
    scored_results = [(enhanced_match_quality(item, title, year), item) for item in unique_results]
    scored_results.sort(reverse=True, key=lambda x: x[0])

    # Log top results
    for i, (score, item) in enumerate(scored_results[:3]):
        if i == 0:
            logger.info(f"Best match: '{item.get('#TITLE')}' with score {score:.2f}")
        else:
            logger.info(f"Alternative match {i}: '{item.get('#TITLE')}' with score {score:.2f}")

    # Apply minimum score threshold
    normalized_threshold = MIN_FUZZ_RATIO * 100
    filtered_results = [(score, item) for score, item in scored_results if score >= normalized_threshold]

    if not filtered_results:
        logger.info(f"No results met minimum threshold of {normalized_threshold:.2f}")
        if scored_results:
            # Return the best match we have even if it's below threshold
            logger.info(f"Using best available match: '{scored_results[0][1].get('#TITLE')}'")
            return [item for _, item in scored_results]
        return []

    # Return items in ranked order
    return [item for _, item in filtered_results]


def is_good_result_set(results, title, year=None):
    """Determine if the results are good enough to stop searching."""
    if not results:
        return False

    # Look for high quality matches
    title_lower = title.lower().strip()

    # Check if we have at least one very good match
    for item in results:
        item_title = item.get("#TITLE", "").lower().strip()

        # Exact match with correct year
        if item_title == title_lower:
            if not year or str(item.get("#YEAR", "")) == str(year):
                return True

        # Main title exact match
        if ":" in item_title:
            main_part = item_title.split(":")[0].strip()
            if main_part == title_lower:
                if not year or str(item.get("#YEAR", "")) == str(year):
                    return True

    # Check if we have enough results overall
    if len(results) >= 10:
        return True

    return False


def search_media(title, year=None, session=None):
    """
    Enhanced search for media with multi-stage fallback and sophisticated ranking.

    Args:
        title: Cleaned title string
        year: Optional year
        session: HTTP session

    Returns:
        Best match results or None
    """
    # First check if this is a known TV show that should override API results
    forced_tv_shows = KNOWN_TV_SHOWS

    title_lower = title.lower()
    title_without_year = re.sub(r'\s*\(\d{4}\)\s*', '', title_lower).strip()

    # If it's a forced TV show, create a synthetic result
    for show in forced_tv_shows:
        if show in title_without_year:
            logger.info(f"OVERRIDE: '{title_without_year}' matches forced TV show '{show}'")
            # Extract year if present
            year_match = re.search(r'\((\d{4})\)', title)
            year = year_match.group(1) if year_match else ""

            # Create a synthetic result
            synthetic_result = {
                "results": [
                    {
                        "title": title_without_year.title(),
                        "year": year,
                        "type": "tv",
                        "imdb_id": "override",
                        "rank": "",
                        "actors": ""
                    }
                ]
            }
            return synthetic_result

    if not title or not session:
        session = create_session()

    # Track all results across multiple query attempts
    all_results = []
    queries_tried = set()

    # STAGE 1: Generate and try primary queries
    primary_queries = generate_primary_queries(title, year)

    for query in primary_queries:
        if query in queries_tried or len(query.strip()) < 2:
            continue

        logger.info(f"Trying query: '{query}'")
        queries_tried.add(query)

        result = search_movie(query, API_URL, session, API_TIMEOUT)

        if result and "description" in result and result["description"]:
            for item in result["description"]:
                # Add the source query for later scoring/analysis
                item_copy = item.copy()
                item_copy["#SOURCE_QUERY"] = query
                all_results.append(item_copy)

        # Respect rate limits - add delay between queries
        if SEARCH_DELAY > 0:
            time.sleep(SEARCH_DELAY)

        # Check if we've reached the maximum number of queries
        if len(queries_tried) >= MAX_SEARCH_QUERIES:
            logger.info(f"Reached maximum search queries limit ({MAX_SEARCH_QUERIES})")
            break

    # STAGE 2: Try title variations if we don't have good results yet
    if not is_good_result_set(all_results, title, year):
        # Generate title variations for problem titles
        title_variations = generate_title_variations(title)

        for variation in title_variations:
            var_queries = generate_primary_queries(variation, year)

            for query in var_queries:
                if query in queries_tried or len(query.strip()) < 2:
                    continue

                logger.info(f"Trying title variation query: '{query}'")
                queries_tried.add(query)

                result = search_movie(query, API_URL, session, API_TIMEOUT)

                if result and "description" in result and result["description"]:
                    for item in result["description"]:
                        item_copy = item.copy()
                        item_copy["#SOURCE_QUERY"] = query
                        all_results.append(item_copy)

                # Respect rate limits and check query limit
                if SEARCH_DELAY > 0:
                    time.sleep(SEARCH_DELAY)

                if len(queries_tried) >= MAX_SEARCH_QUERIES:
                    break

            # If we get good results from a variation, we can stop
            if is_good_result_set(all_results, variation, year):
                logger.info(f"Found good results using title variation: '{variation}'")
                break

    # STAGE 3: Try alternative queries if still needed
    if not is_good_result_set(all_results, title, year) and len(all_results) < 5:
        alt_queries = generate_alternative_queries(title, year)

        for query in alt_queries:
            if query in queries_tried or len(query.strip()) < 2:
                continue

            logger.info(f"Trying alternative query: '{query}'")
            queries_tried.add(query)

            result = search_movie(query, API_URL, session, API_TIMEOUT)

            if result and "description" in result and result["description"]:
                for item in result["description"]:
                    item_copy = item.copy()
                    item_copy["#SOURCE_QUERY"] = query
                    all_results.append(item_copy)

            # Respect rate limits and check query limit
            if SEARCH_DELAY > 0:
                time.sleep(SEARCH_DELAY)

            if len(queries_tried) >= MAX_SEARCH_QUERIES:
                break

    # Try a search on TMDb as a fallback if enabled
    if not all_results and TMDB_ENABLED and TMDB_API_KEY:
        logger.info("Trying TMDb as fallback source")
        tmdb_result = search_tmdb(title, year, session)

        if tmdb_result and "results" in tmdb_result and tmdb_result["results"]:
            # Convert TMDb results to our format
            for item in tmdb_result["results"]:
                tmdb_item = {
                    "#TITLE": item.get("title", "Unknown"),
                    "#YEAR": item.get("year", ""),
                    "#TYPE": item.get("type", "movie"),
                    "#IMDB_ID": f"tmdb_{item.get('id', '')}",
                    "#RANK": "",
                    "#ACTORS": "",
                    "#SOURCE_QUERY": "tmdb"
                }
                all_results.append(tmdb_item)

    # Final processing: deduplicate, score, and rank results
    if not all_results:
        logger.info(f"No results found for '{title}'")
        return None

    ranked_results = process_search_results(all_results, title, year)

    return {"description": ranked_results} if ranked_results else None


def search_imdb(title, original_title):
    """Enhanced IMDb search with better TV show detection and advanced matching"""
    # Clean and prepare the title
    cleaned_title = clean_title_for_search(title)
    logger.info(f"Searching IMDb for: {cleaned_title}")

    # Extract year if present
    year_match = re.search(r'\((\d{4})\)', title)
    year = year_match.group(1) if year_match else None

    # Create HTTP session
    session = create_session()

    # Use enhanced search function
    result = search_media(cleaned_title, year, session)

    # If we get no results, try with the original title
    if not result or not result.get("description"):
        logger.info(f"No results found for cleaned title. Trying original: {original_title}")
        original_cleaned = clean_title_for_search(original_title)
        if original_cleaned != cleaned_title:
            result = search_media(original_cleaned, year, session)

    # Try TV-specific searches if we don't have a TV show yet
    if result and "description" in result and result["description"]:
        found_tv = any(item.get("#TYPE", "").lower() == "tv" for item in result["description"])

        if not found_tv:
            # Check in the original title for TV show indicators
            has_tv_indicator = any(indicator.lower() in original_title.lower() for indicator in TV_INDICATORS)
            is_known_tv = any(show in cleaned_title.lower() for show in KNOWN_TV_SHOWS)

            if has_tv_indicator or is_known_tv:
                logger.info(f"Potentially missed TV show: {cleaned_title}")
                tv_search = f"{cleaned_title} tv series"
                logger.info(f"Trying explicit TV search: '{tv_search}'")
                tv_result = search_media(tv_search, year, session)

                if tv_result and "description" in tv_result and tv_result["description"]:
                    tv_items = [item for item in tv_result["description"] if item.get("#TYPE", "").lower() == "tv"]
                    if tv_items:
                        logger.info(f"Found TV result with explicit search")
                        return {"description": tv_items}

    # Final TMDb check if all else fails
    if not result and TMDB_ENABLED and TMDB_API_KEY:
        logger.info("No IMDb results, trying TMDb directly.")
        tmdb_result = search_tmdb(cleaned_title, year, session)

        if tmdb_result and "results" in tmdb_result and tmdb_result["results"]:
            # Convert to the format expected by the rest of the code
            description = []
            for item in tmdb_result["results"]:
                converted_item = {
                    "#TITLE": item.get("title", "Unknown"),
                    "#YEAR": item.get("year", ""),
                    "#TYPE": item.get("type", "movie").upper(),
                    "#IMDB_ID": f"tmdb_{item.get('id', '')}",
                    "#VOTE_COUNT": item.get("vote_count", 0),
                    "#SOURCE_QUERY": "tmdb_direct"
                }
                description.append(converted_item)

            result = {"description": description}

    return result


def convert_api_response(api_results):
    """Convert the API response format with enhanced TV detection"""
    if not api_results:
        return None

    # Create a new data structure that matches what our script expects
    converted_results = {
        "results": []
    }

    for item in api_results:
        # Extract the fields we need
        title = item.get("#TITLE", "Unknown")
        year = item.get("#YEAR", "")
        imdb_id = item.get("#IMDB_ID", "")
        rank = item.get("#RANK", "")
        actors = item.get("#ACTORS", "")
        aka = item.get("#AKA", "")
        genre = item.get("#GENRE", "")

        # Determine content type using multiple detection methods
        is_tv = False

        # Method 1: Check title for TV show indicators
        if any(indicator.lower() in title.lower() for indicator in TV_INDICATORS):
            is_tv = True
            logger.info(f"TV indicator found in title: {title}")

        # Method 2: Look for season/episode pattern in AKA field if present
        tv_aka_indicators = ["s01", "season 1", "episode", "series", "show"]
        if isinstance(aka, str) and any(pattern.lower() in aka.lower() for pattern in tv_aka_indicators):
            is_tv = True
            logger.info(f"TV indicator found in AKA: {aka}")

        # Method 3: If there's genre info, check for TV-specific genres
        tv_genres = ["tv series", "reality-tv", "talk-show", "game-show", "documentary", "mini-series"]
        if isinstance(genre, str) and any(g.lower() in genre.lower() for g in tv_genres):
            is_tv = True
            logger.info(f"TV indicator found in genre: {genre}")

        # Method 4: Check known TV shows by title
        if any(show.lower() in title.lower() for show in KNOWN_TV_SHOWS):
            is_tv = True
            logger.info(f"Known TV show detected: {title}")

        # Method 5: Check if #TYPE is already set to TV
        if item.get("#TYPE", "").lower() in ["tv", "series", "show"]:
            is_tv = True
            logger.info(f"Type already identified as TV: {title}")

        # Set the content type based on our detection
        content_type = "tv" if is_tv else "movie"

        # Create a result item
        result_item = {
            "title": title,
            "year": year,
            "type": content_type,
            "imdb_id": imdb_id,
            "rank": rank,
            "actors": actors
        }

        converted_results["results"].append(result_item)

    return converted_results


def normalize_title(title, language=FALLBACK_LANGUAGE):
    """
    Normalize title for consistent comparison by removing accents,
    standardizing unicode characters, and handling language-specific issues.
    """
    if not title:
        return ""

    # Handle non-ASCII characters
    normalized = unicodedata.normalize('NFKD', title)
    # Strip accents
    normalized = ''.join([c for c in normalized if not unicodedata.combining(c)])
    # Lowercase
    normalized = normalized.lower()
    # Replace multiple spaces with single space
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    # Language-specific normalization
    if language == 'en':
        # Remove leading articles for English
        normalized = re.sub(r'^(the|a|an) ', '', normalized)
    elif language == 'es':
        # Remove leading articles for Spanish
        normalized = re.sub(r'^(el|la|los|las|un|una|unos|unas) ', '', normalized)
    elif language == 'fr':
        # Remove leading articles for French
        normalized = re.sub(r'^(le|la|les|un|une|des) ', '', normalized)

    return normalized.strip()


def is_collection(folder_name):
    """
    Determine if a folder represents a movie collection (trilogy, anthology, etc.)
    """
    folder_lower = folder_name.lower()

    # Check for explicit collection identifiers
    if any(identifier in folder_lower for identifier in COLLECTION_IDENTIFIERS):
        logger.info(f"Detected collection indicator in '{folder_name}'")
        return True

    # Check for multiple year ranges like "Movie Name (1999-2023)"
    if re.search(r'\((?:19|20)\d{2}-(?:19|20)\d{2}\)', folder_name):
        logger.info(f"Detected year range in '{folder_name}'")
        return True

    # Check for multiple years like "Movie 1998 2001 2003"
    year_matches = re.findall(r'\b(?:19|20)\d{2}\b', folder_name)
    if len(year_matches) >= 2 and len(set(year_matches)) >= 2:
        logger.info(f"Detected multiple years in '{folder_name}': {year_matches}")
        return True

    # Check for patterns like "Movie 1, 2, 3" or "Movie I, II, III"
    if (re.search(r'(?:^|\s)(\d+)(?:,\s+(\d+))+', folder_name) or
            re.search(r'(?:^|\s)([IVX]+)(?:,\s+([IVX]+))+', folder_name)):
        logger.info(f"Detected numbered sequence in '{folder_name}'")
        return True

    # Check for "Part 1 & 2" or similar patterns
    if re.search(r'part\s+\d+\s*(?:&|and)\s*(?:part\s+)?\d+', folder_lower):
        logger.info(f"Detected 'Part X & Y' pattern in '{folder_name}'")
        return True

    return False


def determine_content_type(imdb_data, original_title):
    """Determine if content is a movie or TV show with enhanced detection"""

    # Check for collections first
    if HANDLE_COLLECTIONS and is_collection(original_title):
        logger.info(f"Detected movie collection: '{original_title}'")
        return "collection", format_collection_title(original_title)

    # Check for exact title matches first - these override everything else
    forced_tv_shows = {
        "clarkson's farm": "tv",
        "the grand tour": "tv",
        "ted lasso": "tv",
        "the last of us": "tv",
        "the bear": "tv",
        "the boys": "tv",
        "house of the dragon": "tv",
        "severance": "tv",
        "succession": "tv",
        "fargo": "tv",
        "true detective": "tv",
        "the office": "tv",
        "game of thrones": "tv",
        "breaking bad": "tv"
    }

    # Check for exact matches in the forced TV shows dict
    title_lower = original_title.lower()
    clean_title_lower = re.sub(r'\s*\(\d{4}\)\s*', '', title_lower).strip()

    for show, content_type in forced_tv_shows.items():
        if show in clean_title_lower:
            logger.info(f"FORCED override: '{clean_title_lower}' matches known TV show '{show}'")
            # Extract years if present
            year_match = re.search(r'\((\d{4})\)', original_title)
            year = year_match.group(1) if year_match else ""
            formatted_title = f"{clean_title_lower.title()} ({year})" if year else clean_title_lower.title()
            return content_type, formatted_title

    # Check for anime-specific terms
    anime_term_match = any(term.lower() in original_title.lower() for term in ANIME_TERMS)
    if anime_term_match and not any(term.lower() in original_title.lower() for term in ["movie", "film"]):
        # Anime with OVA/Special/etc. is likely a TV show
        logger.info(f"Detected anime series from terminology: {original_title}")

        # Try to format the title properly
        year_match = re.search(r'\((\d{4})\)', original_title)
        if year_match:
            year = year_match.group(1)
            # Extract the base title (before any season/episode info)
            base_title = re.sub(r'(?i)\s+(?:season|s\d+|ova|special|part|cour).*$', '',
                                re.sub(r'\s*\(\d{4}\)\s*', '', original_title)).strip()
            formatted_title = f"{base_title} ({year})"
        else:
            # No year found, just clean up the title
            base_title = re.sub(r'(?i)\s+(?:season|s\d+|ova|special|part|cour).*$', '', original_title).strip()
            formatted_title = base_title

        return "tv", formatted_title

    # Look for specific TV show indicators in the folder name
    has_tv_indicator = any(indicator.lower() in title_lower for indicator in TV_INDICATORS)
    is_known_tv = any(show in title_lower for show in KNOWN_TV_SHOWS)

    # If we have direct evidence from the name, use it immediately
    if is_known_tv or has_tv_indicator:
        logger.info(
            f"Determined TV show from name patterns: {'Known series' if is_known_tv else 'TV indicators found'}")

        # If we directly extracted a title that's already in "Title (Year)" format
        year_match = re.search(r'(.*)\((\d{4})\)', original_title)
        if year_match:
            title = year_match.group(1).strip()
            year = year_match.group(2)
            formatted_title = f"{title} ({year})"
        else:
            # Extract year from anywhere in the filename if present
            year_anywhere = re.search(r'\b(19\d{2}|20\d{2})\b', original_title)
            year = year_anywhere.group(1) if year_anywhere else ""

            # Strip technical info for a cleaner title
            base_title = re.sub(r'(?:720p|1080p|2160p|HDTV|BluRay|WEB-DL|x264|x265).*$', '', original_title,
                                flags=re.IGNORECASE)
            base_title = re.sub(r'\([^\)]*(?:1080p|720p|HDTV|BluRay|WEB-DL)[^\)]*\).*$', '', base_title,
                                flags=re.IGNORECASE)

            # For TV shows, try to extract the series title without season info
            series_title_match = re.search(r'^(.*?)(?:season|\s+s\d+|\s+e\d+|complete|\()', base_title, re.IGNORECASE)
            if series_title_match:
                base_title = series_title_match.group(1).strip()

            # Clean up and format
            base_title = re.sub(r'[.\-_]', ' ', base_title)
            base_title = re.sub(r'\s+', ' ', base_title).strip()

            # Apply final formatting
            formatted_title = f"{base_title} ({year})" if year else base_title

        logger.info(f"Determined TV show - '{formatted_title}'")
        return "tv", formatted_title

    # If we don't have direct evidence from the name, check the API data
    if not imdb_data or "description" not in imdb_data or not imdb_data["description"]:
        logger.info(f"API search failed for '{original_title}', determining content type from folder name patterns")

        # If we get here, we don't have direct TV evidence and API failed
        # Fall back to regex cleaning and format detection
        year_match = re.search(r'(.*)\((\d{4})\)', original_title)
        if year_match:
            title = year_match.group(1).strip()
            year = year_match.group(2)
            formatted_title = f"{title} ({year})"
        else:
            # Extract year from anywhere in the filename if present
            year_anywhere = re.search(r'\b(19\d{2}|20\d{2})\b', original_title)
            year = year_anywhere.group(1) if year_anywhere else ""

            # Strip technical info for a cleaner title
            base_title = re.sub(r'(?:720p|1080p|2160p|HDTV|BluRay|WEB-DL|x264|x265).*$', '', original_title,
                                flags=re.IGNORECASE)
            base_title = re.sub(r'\([^\)]*(?:1080p|720p|HDTV|BluRay|WEB-DL)[^\)]*\).*$', '', base_title,
                                flags=re.IGNORECASE)

            # Clean up and format
            base_title = re.sub(r'[.\-_]', ' ', base_title)
            base_title = re.sub(r'\s+', ' ', base_title).strip()

            # Apply final formatting
            formatted_title = f"{base_title} ({year})" if year else base_title

        # Check if the cleaned title matches known TV shows
        clean_title_lower = formatted_title.lower()
        if any(show in clean_title_lower for show in KNOWN_TV_SHOWS):
            logger.info(f"Detected known TV show in cleaned title: {formatted_title}")
            return "tv", formatted_title

        # Additional check for TV show patterns in the clean title
        if any(indicator.lower() in clean_title_lower for indicator in TV_INDICATORS):
            logger.info(f"Detected TV indicator in cleaned title: {formatted_title}")
            return "tv", formatted_title

        # Fallback: check if it's a movie collection
        if HANDLE_COLLECTIONS and is_collection(original_title):
            logger.info(f"Detected movie collection as fallback: '{original_title}'")
            return "collection", format_collection_title(original_title)

        # Default to movie if we can't determine from name patterns
        logger.info(f"Defaulting to movie classification: '{formatted_title}'")
        return "movie", formatted_title

    # Regular API-based detection using the first result
    result = imdb_data["description"][0]

    # Extract content type and year
    content_type = result.get("#TYPE", "movie").lower()  # Default to movie if not specified
    year = result.get("#YEAR", "")
    title = result.get("#TITLE", "")

    # Create formatted title with year
    formatted_title = f"{title} ({year})" if year else title

    # If the API doesn't return a type or it's unknown, try to guess from the original title
    if content_type not in ["movie", "tv", "series", "show"]:
        # Check for common TV show indicators
        if any(indicator.lower() in original_title.lower() for indicator in TV_INDICATORS):
            content_type = "tv"
        # Also check if the title matches known TV shows
        elif any(show in title.lower() for show in KNOWN_TV_SHOWS):
            content_type = "tv"
        else:
            content_type = "movie"

    # Normalize content type
    if content_type in ["tv", "series", "show"]:
        content_type = "tv"
    else:
        content_type = "movie"

    # Override for collections
    if HANDLE_COLLECTIONS and is_collection(original_title):
        logger.info(f"Overriding API result for collection: '{original_title}'")
        return "collection", format_collection_title(original_title)

    logger.info(f"API determined content type: {content_type}")
    return content_type, formatted_title


def format_collection_title(folder_name):
    """Format a collection title properly"""
    # Try to extract range of years if present
    year_range_match = re.search(r'\((\d{4})-(\d{4})\)', folder_name)
    if year_range_match:
        base_title = re.sub(r'\s*\(\d{4}-\d{4}\)\s*', '', folder_name).strip()
        start_year = year_range_match.group(1)
        end_year = year_range_match.group(2)
        return f"{base_title} Collection ({start_year}-{end_year})"

    # Try to extract multiple individual years
    years = re.findall(r'\b(19\d{2}|20\d{2})\b', folder_name)
    if len(years) >= 2:
        # Remove years from the title
        base_title = re.sub(r'\b(19\d{2}|20\d{2})\b', '', folder_name).strip()
        # Remove any parentheses left behind
        base_title = re.sub(r'\(\s*\)', '', base_title).strip()
        # Clean up spaces and punctuation
        base_title = re.sub(r'\s+', ' ', base_title).strip()
        # Remove trailing hyphens or commas
        base_title = re.sub(r'[-,]+$', '', base_title).strip()

        # Check if "Collection" or "Trilogy" etc. is already in the title
        has_collection_word = any(word in base_title.lower() for word in COLLECTION_IDENTIFIERS)

        if not has_collection_word:
            # Determine if it's a trilogy, collection, etc.
            if len(years) == 3:
                base_title = f"{base_title} Trilogy"
            elif len(years) == 2:
                base_title = f"{base_title} Collection"
            elif len(years) == 4:
                base_title = f"{base_title} Quadrilogy"
            else:
                base_title = f"{base_title} Collection"

        # Format with year range
        min_year = min(years)
        max_year = max(years)
        if min_year != max_year:
            return f"{base_title} ({min_year}-{max_year})"
        else:
            return f"{base_title} ({min_year})"

    # If no years found, just clean up the title and add "Collection"
    base_title = re.sub(r'[.\-_]', ' ', folder_name).strip()
    has_collection_word = any(word in base_title.lower() for word in COLLECTION_IDENTIFIERS)

    if not has_collection_word:
        return f"{base_title} Collection"
    else:
        return base_title


def detect_tv_seasons(folder_path):
    """
    Detect season folders and special content folders within a TV show folder.
    Returns a tuple of (seasons, special_folders)
    """
    seasons = []
    special_folders = []

    # Common patterns for season folders
    season_patterns = [
        r"[Ss]eason\s*(\d+)",
        r"[Ss](\d+)",
        r"Season.(\d+)",
        r"S(\d+)E\d+"
    ]

    try:
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)

            if os.path.isdir(item_path):
                # Check if folder name matches any season pattern
                is_season = False
                for pattern in season_patterns:
                    match = re.search(pattern, item)
                    if match:
                        season_num = match.group(1)
                        seasons.append((item_path, int(season_num)))
                        is_season = True
                        break

                # If not a season, check if it's a special content folder
                if not is_season:
                    is_special = False
                    for special_pattern in SPECIAL_CONTENT_FOLDERS:
                        if special_pattern.lower() in item.lower():
                            special_folders.append((item_path, item))
                            is_special = True
                            break
    except OSError as e:
        logger.error(f"Error detecting seasons in '{folder_path}': {str(e)}")

    return seasons, special_folders


def detect_season_from_name(folder_name):
    """Extract season number from folder name with enhanced detection"""
    # Look for patterns like "Season 2", "S02", etc.
    season_patterns = [
        r'Season\s*(\d+)',
        r'\bS(\d{1,2})\b',
        r'S(\d{1,2})E\d+',
        r'Season.(\d+)',
        r'Saison\s*(\d+)',  # French
        r'Staffel\s*(\d+)',  # German
        r'Temporada\s*(\d+)',  # Spanish/Portuguese
        r'Series\s*(\d+)',  # British terminology
    ]

    for pattern in season_patterns:
        match = re.search(pattern, folder_name, re.IGNORECASE)
        if match:
            try:
                season_num = int(match.group(1))
                logger.info(f"Detected Season {season_num} from folder name")
                return season_num
            except (ValueError, IndexError):
                continue

    # Check for anime-specific patterns
    anime_patterns = [
        r'S(\d+)P(\d+)',  # Season+Part format
        r'Part\s*(\d+)',  # Anime part numbering
        r'Cour\s*(\d+)',  # Anime cour numbering
    ]

    for pattern in anime_patterns:
        match = re.search(pattern, folder_name, re.IGNORECASE)
        if match:
            try:
                season_num = int(match.group(1))
                logger.info(f"Detected Anime Season/Part {season_num} from folder name")
                return season_num
            except (ValueError, IndexError):
                continue

    # Look for special/OVA markers (treating as season 0)
    special_patterns = [
        r'\bOVA\b',
        r'\bSpecials?\b',
        r'\bSP\b',
        r'\bEpisode\s+0\b',
        r'\bS00\b',
        r'\bSeason\s+0\b',
    ]

    for pattern in special_patterns:
        if re.search(pattern, folder_name, re.IGNORECASE):
            logger.info(f"Detected Specials/Season 0 from folder name")
            return 0

    # Check for multi-season pattern (like S01-S03)
    multi_season = re.search(r'S(\d+)[^\d]?[-~]S?(\d+)', folder_name, re.IGNORECASE)
    if multi_season:
        # For multi-season packages, we need special handling
        try:
            start_season = int(multi_season.group(1))
            end_season = int(multi_season.group(2))
            logger.info(f"Detected multi-season package {start_season}-{end_season} from folder name")
            # Return a tuple instead of an integer to indicate multiple seasons
            return (start_season, end_season)
        except (ValueError, IndexError):
            pass

    return None


def detect_episode_files(folder_path):
    """
    Detect if a folder contains episode files directly (not in season subfolders).
    Returns a list of episode files if found.
    """
    episode_files = []

    # Regex patterns to identify TV episodes
    episode_patterns = [
        r'S\d+E\d+',  # S01E01 format
        r'\d+x\d+',  # 1x01 format
        r'Episode\s*\d+',  # Episode 1
        r'E\d+',  # E01 format
        r'Ep\d+'  # Ep01 format
    ]

    try:
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)

            # Check only files with media extensions
            if os.path.isfile(item_path) and any(item.lower().endswith(ext) for ext in MEDIA_EXTENSIONS):
                # Check if file matches any episode pattern
                if any(re.search(pattern, item, re.IGNORECASE) for pattern in episode_patterns):
                    episode_files.append((item_path, item))

        if episode_files:
            logger.info(f"Found {len(episode_files)} episode files directly in {folder_path}")

        return episode_files
    except OSError as e:
        logger.error(f"Error detecting episode files in '{folder_path}': {str(e)}")
        return []


def find_existing_tv_series(formatted_title):
    """Look for existing TV series in the TV directory that might match"""
    # Extract the title without year for flexible matching
    base_title = re.sub(r'\s*\(\d{4}\)$', '', formatted_title).strip()

    # Normalize for comparison
    normalized_title = normalize_title(base_title)

    try:
        for item in os.listdir(TV_DIR):
            item_path = os.path.join(TV_DIR, item)

            if os.path.isdir(item_path):
                # Extract base title without year for comparison
                item_base = re.sub(r'\s*\(\d{4}\)$', '', item).strip()
                normalized_item = normalize_title(item_base)

                # Check if normalized titles match
                if normalized_item == normalized_title:
                    logger.info(f"Found existing TV series: {item}")
                    return item_path, item

                # Use fuzzy matching for more flexibility
                if fuzz.ratio(normalized_item, normalized_title) > 90:
                    logger.info(f"Found similar existing TV series: {item} (fuzzy match)")
                    return item_path, item
    except OSError as e:
        logger.error(f"Error searching for existing TV series: {str(e)}")

    return None, None


def handle_multi_part_content(folder_path, destination_dir, title, year=None):
    """
    Special handling for content that has multiple parts (like "Part 1 & 2")
    """
    # Extract clean title and ensure it has "Collection" in the name
    title_without_year = re.sub(r'\s*\(\d{4}(?:-\d{4})?\)\s*', '', title).strip()

    # Make sure it ends with "Collection" if not already specified
    if not any(identifier.lower() in title_without_year.lower() for identifier in COLLECTION_IDENTIFIERS):
        collection_title = f"{title_without_year} Collection"
    else:
        collection_title = title_without_year

    # Format the year range if we have years
    if year:
        formatted_title = f"{collection_title} ({year})"
    else:
        # Try to extract years from the original path
        folder_name = os.path.basename(folder_path)
        years = re.findall(r'\b(19\d{2}|20\d{2})\b', folder_name)
        if len(years) >= 2:
            min_year = min(years)
            max_year = max(years)
            if min_year != max_year:
                formatted_title = f"{collection_title} ({min_year}-{max_year})"
            else:
                formatted_title = f"{collection_title} ({min_year})"
        else:
            formatted_title = collection_title

    # Create the final destination path
    dest_path = os.path.join(destination_dir, formatted_title)

    # Check if the destination already exists
    if os.path.exists(dest_path):
        count = 1
        while True:
            alt_path = f"{dest_path} ({count})"
            if not os.path.exists(alt_path):
                dest_path = alt_path
                break
            count += 1

    logger.info(f"Multi-part content will be moved to: {dest_path}")

    # Move the folder
    try:
        if not DRY_RUN:
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.move(folder_path, dest_path)
            logger.info(f"Successfully moved multi-part content to: {dest_path}")
        else:
            logger.info(f"[DRY RUN] Would move '{folder_path}' to '{dest_path}'")
        return True
    except Exception as e:
        logger.error(f"Error moving multi-part content: {str(e)}")
        return False


def handle_mixed_content(folder_path, mixed_types):
    """Handle folders that contain a mix of movies and TV shows"""
    logger.info(f"Handling mixed content folder: {folder_path}")

    # Create a subfolder in each destination for this mixed content
    source_folder_name = os.path.basename(folder_path)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    mixed_folder_name = f"{source_folder_name}_mixed_{timestamp}"

    # Process each item
    success = True
    for item_path, content_type, title in mixed_types:
        if content_type == "movie":
            dest_dir = MOVIES_DIR
        elif content_type == "tv":
            dest_dir = TV_DIR
        elif content_type == "collection":
            dest_dir = COLLECTIONS_DIR
        else:
            logger.warning(f"Unknown content type '{content_type}' for '{item_path}'")
            continue

        mixed_dest_path = os.path.join(dest_dir, mixed_folder_name, title)

        try:
            if not DRY_RUN:
                os.makedirs(os.path.dirname(mixed_dest_path), exist_ok=True)
                logger.info(f"Moving mixed content: {item_path} → {mixed_dest_path}")
                shutil.move(item_path, mixed_dest_path)
            else:
                logger.info(f"[DRY RUN] Would move '{item_path}' to '{mixed_dest_path}'")
        except Exception as e:
            logger.error(f"Error moving mixed content item: {str(e)}")
            success = False

    # If all items were moved successfully and the source folder is empty, remove it
    if success and not DRY_RUN:
        try:
            if not os.listdir(folder_path):
                os.rmdir(folder_path)
                logger.info(f"Removed empty source folder: {folder_path}")
        except Exception as e:
            logger.error(f"Error removing empty source folder: {str(e)}")

    return success


def detect_movie_version(folder_name):
    """
    Detect if a movie has a specific version (Director's Cut, Extended, etc.)
    Returns the detected version or None
    """
    folder_lower = folder_name.lower()

    for version in MOVIE_VERSION_INDICATORS:
        if version.lower() in folder_lower:
            logger.info(f"Detected movie version: {version}")
            return version

    return None


def organize_collection_content(collection_path):
    """
    Organize the internal structure of a movie collection folder
    by formatting individual movie titles within it
    """
    collection_name = os.path.basename(collection_path)
    logger.info(f"Organizing internal structure of collection: {collection_name}")

    # Extract the collection base name without years
    collection_base = re.sub(r'\s*\(\d{4}(?:-\d{4})?\)\s*', '', collection_name).strip()

    # If this is a well-known collection, use its entry in the database for naming consistency
    if collection_base.lower() in CONTENT_DATABASE["collections"]:
        collection_info = CONTENT_DATABASE["collections"][collection_base.lower()]
        logger.info(f"Using known collection information for {collection_base}")

    try:
        # Check each subfolder/file in the collection
        for item in os.listdir(collection_path):
            item_path = os.path.join(collection_path, item)

            # Only process directories
            if os.path.isdir(item_path):
                # Try to extract the movie title and clean it
                cleaned_title = clean_folder_name(item)
                logger.info(f"Cleaned collection item title: {cleaned_title}")

                # Check if this looks like a proper movie name already
                if re.search(r'.*\(\d{4}\)', cleaned_title):
                    # If it's already properly formatted, just ensure it follows collection naming
                    continue

                # If it's not well-formatted, we'll need to make our best guess at the proper title
                # This is a simplification - in a real implementation you'd add more logic here
                # to extract the correct title for each movie in the collection
    except OSError as e:
        logger.error(f"Error organizing collection content: {str(e)}")

    # Return True if we've made changes, False otherwise
    return True


def handle_nested_content(folder_path, max_depth=3):
    """
    Handle deeply nested folder structures by recursively looking for media content
    Returns a list of (path, content_type, title) tuples for all found media
    """
    if max_depth <= 0:
        logger.warning(f"Reached maximum recursion depth in {folder_path}")
        return []

    found_media = []

    try:
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)

            # Only process directories
            if os.path.isdir(item_path):
                # Check if this directory looks like it contains media
                if any(media_indicator in item.lower() for media_indicator in
                       ['movie', 'film', 'tv', 'series', 'season', 'show']):
                    # This folder might be a media container - process it
                    logger.info(f"Found potential media folder: {item}")

                    # Clean the folder name to get a proper title
                    cleaned_title = clean_folder_name(item)

                    # Search for metadata to determine content type
                    imdb_data = search_imdb(cleaned_title, item)
                    content_type, formatted_title = determine_content_type(imdb_data, item)

                    found_media.append((item_path, content_type, formatted_title))
                else:
                    # This might be an intermediate folder - recurse into it
                    nested_media = handle_nested_content(item_path, max_depth - 1)
                    found_media.extend(nested_media)
    except OSError as e:
        logger.error(f"Error inspecting nested content in {folder_path}: {str(e)}")

    return found_media


def process_folder(folder_path):
    """Enhanced process_folder with comprehensive edge case handling"""
    # Get folder name and initialize logging
    folder_name = os.path.basename(folder_path)
    logger.info(f"\n{'=' * 80}\nProcessing: {folder_name}\n{'=' * 80}")

    # Skip if this is a Windows System Volume Information folder
    if folder_name == "System Volume Information":
        logger.info("Skipping System Volume Information folder")
        return

    # Check if this might be deeply nested content that needs special handling
    if any(len(name) < 5 or name in ["disk1", "disk2", "disk", "cd1", "cd2", "dvd1", "dvd2"]
           for name in folder_name.lower().split()):
        logger.info(f"Detected potential nested content structure: {folder_name}")
        nested_content = handle_nested_content(folder_path, MAX_SCAN_DEPTH)

        if nested_content:
            # Check if we have mixed content (both movies and TV shows)
            content_types = set(item[1] for item in nested_content)
            if len(content_types) > 1:
                logger.info(f"Detected mixed content types in nested structure: {content_types}")
                handle_mixed_content(folder_path, nested_content)
                return

            # If all content is of the same type, process according to that type
            first_item = nested_content[0]
            if first_item[1] == "movie":
                logger.info("All nested content appears to be movies")
                # Process as a movie collection
                collection_title = format_collection_title(folder_name)
                dest_path = os.path.join(COLLECTIONS_DIR, collection_title)

                # Move the entire folder to the collections directory
                try:
                    if not DRY_RUN:
                        os.makedirs(COLLECTIONS_DIR, exist_ok=True)
                        shutil.move(folder_path, dest_path)
                        logger.info(f"Moved nested movie collection to: {dest_path}")
                    else:
                        logger.info(f"[DRY RUN] Would move nested movie collection to: {dest_path}")
                except Exception as e:
                    logger.error(f"Error moving nested movie collection: {str(e)}")
                return

            elif first_item[1] == "tv":
                logger.info("All nested content appears to be from the same TV show")
                # Process as a TV show with nested season structure
                tv_title = first_item[2]  # Use the formatted title from the first item
                dest_path = os.path.join(TV_DIR, tv_title)

                # Move the entire folder to the TV directory
                try:
                    if not DRY_RUN:
                        os.makedirs(TV_DIR, exist_ok=True)
                        shutil.move(folder_path, dest_path)
                        logger.info(f"Moved nested TV show to: {dest_path}")
                    else:
                        logger.info(f"[DRY RUN] Would move nested TV show to: {dest_path}")
                except Exception as e:
                    logger.error(f"Error moving nested TV show: {str(e)}")
                return

            # Return since we've handled the nested content
            return

    # Get corrected title using Ollama or direct extraction
    try:
        corrected_title = clean_folder_name(folder_name)
        logger.info(f"Corrected title: {corrected_title}")
    except Exception as e:
        logger.error(f"Error cleaning folder name: {str(e)}")
        # Use regex cleaning as fallback
        corrected_title = regex_clean_title(folder_name)
        logger.info(f"Fallback corrected title: {corrected_title}")

    # Detect season from folder name for TV shows
    detected_season = detect_season_from_name(folder_name)
    if detected_season:
        if isinstance(detected_season, tuple):
            logger.info(f"This appears to be a multi-season package: Seasons {detected_season[0]}-{detected_season[1]}")
        else:
            logger.info(f"This appears to be Season {detected_season}")

    # Check for direct episode files (for TV shows without season folders)
    episode_files = detect_episode_files(folder_path)
    has_episode_files = len(episode_files) > 0

    # Search IMDB for the title
    imdb_data = search_imdb(corrected_title, folder_name)

    # Determine content type and get formatted title
    content_type, formatted_title = determine_content_type(imdb_data, folder_name)

    if not formatted_title:
        logger.warning(f"Couldn't determine a proper title for '{folder_name}', skipping.")
        return

    logger.info(f"Found content: {formatted_title} (Type: {content_type})")

    # Check for movie versions if this is a movie
    movie_version = None
    if content_type == "movie":
        movie_version = detect_movie_version(folder_name)
        if movie_version:
            # Append the version to the formatted title if not already included
            if movie_version.lower() not in formatted_title.lower():
                # Extract title and year
                title_year_match = re.match(r'(.*?)(\s*\(\d{4}\))', formatted_title)
                if title_year_match:
                    base_title = title_year_match.group(1)
                    year_part = title_year_match.group(2)
                    formatted_title = f"{base_title} - {movie_version}{year_part}"
                else:
                    formatted_title = f"{formatted_title} - {movie_version}"
                logger.info(f"Updated title with version: {formatted_title}")

    # Process based on content type
    if content_type == "movie":
        # It's a movie, move to movies directory
        dest_path = os.path.join(MOVIES_DIR, formatted_title)

        # Check if destination already exists
        if os.path.exists(dest_path):
            logger.warning(f"Destination already exists: {dest_path}")
            # In case of conflict, add a unique identifier
            count = 1
            while True:
                alt_dest = f"{dest_path} ({count})"
                if not os.path.exists(alt_dest):
                    dest_path = alt_dest
                    logger.info(f"Using alternative destination: {dest_path}")
                    break
                count += 1

        # Move the directory
        logger.info(f"Moving to: {dest_path}")
        try:
            if not DRY_RUN:
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.move(folder_path, dest_path)
                logger.info(f"Successfully moved to: {dest_path}")

                # Add to database for future reference
                base_title = re.sub(r'\s*\(\d{4}\)\s*', '', formatted_title).lower()
                CONTENT_DATABASE["movies"][base_title] = {
                    "title": formatted_title,
                    "year": re.search(r'\((\d{4})\)', formatted_title).group(1) if re.search(r'\(\d{4}\)',
                                                                                             formatted_title) else "",
                    "path": dest_path,
                    "version": movie_version
                }
            else:
                logger.info(f"[DRY RUN] Would move '{folder_path}' to '{dest_path}'")
        except (OSError, shutil.Error) as e:
            logger.error(f"Error moving '{folder_name}': {str(e)}")

    elif content_type == "tv":
        # Handle TV show with enhanced edge case management
        tv_show_path = os.path.join(TV_DIR, formatted_title)
        title_for_season = formatted_title

        # Try to find existing TV series that might match (for consistency)
        existing_path, existing_title = find_existing_tv_series(formatted_title)
        if existing_path:
            logger.info(f"Will use existing TV show folder: {existing_title}")
            tv_show_path = existing_path
            title_for_season = existing_title

        # If we have a season detected from the folder name, use it
        if detected_season:
            # Handle multi-season package
            if isinstance(detected_season, tuple):
                start_season, end_season = detected_season
                logger.info(f"Processing multi-season package: Seasons {start_season}-{end_season}")

                # Create TV show path if it doesn't exist
                if not DRY_RUN:
                    os.makedirs(tv_show_path, exist_ok=True)
                    logger.info(f"Created/verified TV show directory: {tv_show_path}")
                else:
                    logger.info(f"[DRY RUN] Would create TV show directory: {tv_show_path}")

                # Create a season folder for each season in the range
                for season_num in range(start_season, end_season + 1):
                    season_folder = f"{title_for_season} Season {season_num}"
                    dest_season_path = os.path.join(tv_show_path, season_folder)

                    # Check if this season folder already exists
                    if os.path.exists(dest_season_path):
                        logger.warning(f"Season folder already exists: {dest_season_path}")
                        continue

                    # Look for content for this specific season
                    season_pattern = rf'(?i)s{season_num:02d}|season\s*{season_num}\b'
                    season_files = []

                    try:
                        for item in os.listdir(folder_path):
                            item_path = os.path.join(folder_path, item)
                            if os.path.isfile(item_path) and re.search(season_pattern, item):
                                season_files.append((item_path, item))
                    except OSError as e:
                        logger.error(f"Error scanning for season {season_num} files: {str(e)}")

                    # If we found files for this season, create the season folder and move them
                    if season_files:
                        logger.info(f"Found {len(season_files)} files for Season {season_num}")

                        if not DRY_RUN:
                            os.makedirs(dest_season_path, exist_ok=True)
                            logger.info(f"Created season directory: {dest_season_path}")

                            # Move the season files
                            for file_path, file_name in season_files:
                                dest_file_path = os.path.join(dest_season_path, file_name)
                                logger.info(f"Moving: {file_name} to {dest_season_path}")
                                shutil.move(file_path, dest_file_path)
                        else:
                            logger.info(f"[DRY RUN] Would create season directory: {dest_season_path}")
                            logger.info(f"[DRY RUN] Would move {len(season_files)} files to {dest_season_path}")

                # After processing all seasons, check if there are files left
                # These might be common files or special features
                try:
                    remaining_files = [f for f in os.listdir(folder_path) if
                                       os.path.isfile(os.path.join(folder_path, f))]
                    if remaining_files:
                        # Create a "Extras" or "Common" folder
                        extras_folder = f"{title_for_season} - Extras"
                        dest_extras_path = os.path.join(tv_show_path, extras_folder)

                        if not DRY_RUN:
                            os.makedirs(dest_extras_path, exist_ok=True)
                            logger.info(f"Created extras directory: {dest_extras_path}")

                            # Move remaining files
                            for file_name in remaining_files:
                                file_path = os.path.join(folder_path, file_name)
                                dest_file_path = os.path.join(dest_extras_path, file_name)
                                logger.info(f"Moving extra file: {file_name} to {dest_extras_path}")
                                shutil.move(file_path, dest_file_path)
                        else:
                            logger.info(f"[DRY RUN] Would create extras directory: {dest_extras_path}")
                            logger.info(f"[DRY RUN] Would move {len(remaining_files)} extra files")
                except OSError as e:
                    logger.error(f"Error processing remaining files: {str(e)}")

                # Check for remaining subdirectories - these might be specials or extras
                try:
                    remaining_dirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
                    for dir_name in remaining_dirs:
                        dir_path = os.path.join(folder_path, dir_name)
                        dest_dir_name = f"{title_for_season} - {dir_name}"
                        dest_dir_path = os.path.join(tv_show_path, dest_dir_name)

                        if not DRY_RUN:
                            logger.info(f"Moving extra directory: {dir_name} to {dest_dir_path}")
                            shutil.move(dir_path, dest_dir_path)
                        else:
                            logger.info(f"[DRY RUN] Would move extra directory: {dir_name} to {dest_dir_path}")
                except OSError as e:
                    logger.error(f"Error processing remaining directories: {str(e)}")

                # Finally, remove the original folder if it's empty
                if not DRY_RUN:
                    try:
                        if not os.listdir(folder_path):
                            os.rmdir(folder_path)
                            logger.info(f"Removed empty source folder: {folder_path}")
                    except OSError as e:
                        logger.error(f"Error removing empty source folder: {str(e)}")
            else:
                # Regular single season handling
                season_folder = f"{title_for_season} Season {detected_season}"
                dest_season_path = os.path.join(tv_show_path, season_folder)

                logger.info(f"Creating season folder: {season_folder}")

                # Create TV show path if it doesn't exist
                try:
                    if not DRY_RUN:
                        os.makedirs(tv_show_path, exist_ok=True)
                        logger.info(f"Created/verified TV show directory: {tv_show_path}")
                    else:
                        logger.info(f"[DRY RUN] Would create TV show directory: {tv_show_path}")

                    # Check if season folder already exists
                    if os.path.exists(dest_season_path):
                        logger.warning(f"Season folder already exists: {dest_season_path}")
                        # Could handle merging here if needed
                        return

                    # Create season folder and move content
                    if not DRY_RUN:
                        os.makedirs(dest_season_path, exist_ok=True)
                        logger.info(f"Created season directory: {dest_season_path}")

                        # Move all content to the season folder
                        logger.info(f"Moving content to: {dest_season_path}")
                        for item in os.listdir(folder_path):
                            item_path = os.path.join(folder_path, item)
                            dest_item_path = os.path.join(dest_season_path, item)
                            logger.info(f"Moving: {item} to {dest_item_path}")
                            shutil.move(item_path, dest_item_path)

                        # Remove the original folder if it's now empty
                        if not os.listdir(folder_path):
                            os.rmdir(folder_path)
                            logger.info(f"Removed empty source folder: {folder_path}")

                        logger.info(f"Successfully moved all content to: {dest_season_path}")

                        # Add to database for future reference
                        base_title = re.sub(r'\s*\(\d{4}\)\s*', '', title_for_season).lower()
                        if base_title not in CONTENT_DATABASE["tv_shows"]:
                            CONTENT_DATABASE["tv_shows"][base_title] = {
                                "title": title_for_season,
                                "year": re.search(r'\((\d{4})\)', title_for_season).group(1) if re.search(r'\(\d{4}\)',
                                                                                                          title_for_season) else "",
                                "path": tv_show_path,
                                "seasons": []
                            }
                        if detected_season not in CONTENT_DATABASE["tv_shows"][base_title]["seasons"]:
                            CONTENT_DATABASE["tv_shows"][base_title]["seasons"].append(detected_season)
                    else:
                        logger.info(f"[DRY RUN] Would create season directory: {dest_season_path}")
                        logger.info(f"[DRY RUN] Would move all content from {folder_path} to {dest_season_path}")
                except (OSError, shutil.Error) as e:
                    logger.error(f"Error moving content to season folder: {str(e)}")
        else:
            # No specific season detected from the folder name

            # Check if there are episode files directly in the folder
            if has_episode_files:
                logger.info(f"Found {len(episode_files)} episode files directly in the folder")

                # Determine the season number from the episodes if possible
                season_from_episodes = None
                for _, episode_name in episode_files:
                    season_match = re.search(r'S(\d+)E\d+', episode_name, re.IGNORECASE)
                    if season_match:
                        season_from_episodes = int(season_match.group(1))
                        break

                if season_from_episodes is not None:
                    # We can determine the season from episodes
                    logger.info(f"Detected Season {season_from_episodes} from episode filenames")

                    # Create the season folder
                    season_folder = f"{title_for_season} Season {season_from_episodes}"
                    dest_season_path = os.path.join(tv_show_path, season_folder)

                    try:
                        if not DRY_RUN:
                            os.makedirs(tv_show_path, exist_ok=True)
                            os.makedirs(dest_season_path, exist_ok=True)
                            logger.info(f"Created season directory: {dest_season_path}")

                            # Move all episode files to the season folder
                            for file_path, file_name in episode_files:
                                dest_file_path = os.path.join(dest_season_path, file_name)
                                logger.info(f"Moving episode: {file_name} to {dest_season_path}")
                                shutil.move(file_path, dest_file_path)

                            # Move any other media files that might be episodes
                            for item in os.listdir(folder_path):
                                item_path = os.path.join(folder_path, item)
                                if os.path.isfile(item_path) and any(
                                        item.lower().endswith(ext) for ext in MEDIA_EXTENSIONS):
                                    dest_item_path = os.path.join(dest_season_path, item)
                                    logger.info(f"Moving additional media file: {item} to {dest_season_path}")
                                    shutil.move(item_path, dest_item_path)

                            # Check for any remaining files/folders - these might be extras
                            remaining_items = os.listdir(folder_path)
                            if remaining_items:
                                extras_folder = f"{title_for_season} - Extras"
                                dest_extras_path = os.path.join(tv_show_path, extras_folder)
                                os.makedirs(dest_extras_path, exist_ok=True)

                                for item in remaining_items:
                                    item_path = os.path.join(folder_path, item)
                                    dest_item_path = os.path.join(dest_extras_path, item)
                                    logger.info(f"Moving extra content: {item} to {dest_extras_path}")
                                    shutil.move(item_path, dest_item_path)

                            # Remove the source folder if it's empty
                            if not os.listdir(folder_path):
                                os.rmdir(folder_path)
                                logger.info(f"Removed empty source folder: {folder_path}")

                            # Update database
                            base_title = re.sub(r'\s*\(\d{4}\)\s*', '', title_for_season).lower()
                            if base_title not in CONTENT_DATABASE["tv_shows"]:
                                CONTENT_DATABASE["tv_shows"][base_title] = {
                                    "title": title_for_season,
                                    "year": re.search(r'\((\d{4})\)', title_for_season).group(1) if re.search(
                                        r'\(\d{4}\)', title_for_season) else "",
                                    "path": tv_show_path,
                                    "seasons": []
                                }
                            if season_from_episodes not in CONTENT_DATABASE["tv_shows"][base_title]["seasons"]:
                                CONTENT_DATABASE["tv_shows"][base_title]["seasons"].append(season_from_episodes)
                        else:
                            logger.info(f"[DRY RUN] Would create season directory: {dest_season_path}")
                            logger.info(f"[DRY RUN] Would move all episode files to the season folder")
                    except (OSError, shutil.Error) as e:
                        logger.error(f"Error processing episode files: {str(e)}")
                else:
                    # Can't determine season from episodes, create a "Season 1" folder
                    logger.info("No explicit season number found in episodes, assuming Season 1")

                    # Create the season folder
                    season_folder = f"{title_for_season} Season 1"
                    dest_season_path = os.path.join(tv_show_path, season_folder)

                    try:
                        if not DRY_RUN:
                            os.makedirs(tv_show_path, exist_ok=True)
                            os.makedirs(dest_season_path, exist_ok=True)
                            logger.info(f"Created season directory: {dest_season_path}")

                            # Move all media files to the season folder
                            for item in os.listdir(folder_path):
                                item_path = os.path.join(folder_path, item)
                                if os.path.isfile(item_path) and any(
                                        item.lower().endswith(ext) for ext in MEDIA_EXTENSIONS):
                                    dest_item_path = os.path.join(dest_season_path, item)
                                    logger.info(f"Moving media file: {item} to {dest_season_path}")
                                    shutil.move(item_path, dest_item_path)

                            # Process any remaining content as extras
                            remaining_items = [item for item in os.listdir(folder_path)]
                            if remaining_items:
                                extras_folder = f"{title_for_season} - Extras"
                                dest_extras_path = os.path.join(tv_show_path, extras_folder)
                                os.makedirs(dest_extras_path, exist_ok=True)

                                for item in remaining_items:
                                    item_path = os.path.join(folder_path, item)
                                    dest_item_path = os.path.join(dest_extras_path, item)
                                    logger.info(f"Moving extra content: {item} to {dest_extras_path}")
                                    shutil.move(item_path, dest_item_path)

                            # Remove the source folder if it's empty
                            if not os.listdir(folder_path):
                                os.rmdir(folder_path)
                                logger.info(f"Removed empty source folder: {folder_path}")

                            # Update database
                            base_title = re.sub(r'\s*\(\d{4}\)\s*', '', title_for_season).lower()
                            if base_title not in CONTENT_DATABASE["tv_shows"]:
                                CONTENT_DATABASE["tv_shows"][base_title] = {
                                    "title": title_for_season,
                                    "year": re.search(r'\((\d{4})\)', title_for_season).group(1) if re.search(
                                        r'\(\d{4}\)', title_for_season) else "",
                                    "path": tv_show_path,
                                    "seasons": [1]
                                }
                        else:
                            logger.info(f"[DRY RUN] Would create season directory: {dest_season_path}")
                            logger.info(f"[DRY RUN] Would move all media files to Season 1 folder")
                    except (OSError, shutil.Error) as e:
                        logger.error(f"Error creating Season 1 folder: {str(e)}")
            else:
                # No specific season detected, check if there are season subdirectories
                seasons, special_folders = detect_tv_seasons(folder_path)

                if seasons:
                    logger.info(f"Detected {len(seasons)} season directories within the folder")
                    try:
                        # Create main TV show folder
                        if not DRY_RUN:
                            os.makedirs(tv_show_path, exist_ok=True)
                            logger.info(f"Created/verified TV show directory: {tv_show_path}")
                        else:
                            logger.info(f"[DRY RUN] Would create TV show directory: {tv_show_path}")

                        # Move each season
                        for season_path, season_num in seasons:
                            # Format the season folder name with show title
                            season_folder_name = f"{title_for_season} Season {season_num}"
                            dest_season_path = os.path.join(tv_show_path, season_folder_name)

                            # Check if destination season already exists
                            if os.path.exists(dest_season_path):
                                logger.warning(f"Season folder already exists: {dest_season_path}")
                                continue

                            # Move the season
                            try:
                                if not DRY_RUN:
                                    logger.info(f"Moving {season_folder_name} to: {dest_season_path}")
                                    shutil.move(season_path, dest_season_path)
                                    logger.info(f"Successfully moved to: {dest_season_path}")

                                    # Update database
                                    base_title = re.sub(r'\s*\(\d{4}\)\s*', '', title_for_season).lower()
                                    if base_title not in CONTENT_DATABASE["tv_shows"]:
                                        CONTENT_DATABASE["tv_shows"][base_title] = {
                                            "title": title_for_season,
                                            "year": re.search(r'\((\d{4})\)', title_for_season).group(1) if re.search(
                                                r'\(\d{4}\)', title_for_season) else "",
                                            "path": tv_show_path,
                                            "seasons": []
                                        }
                                    if season_num not in CONTENT_DATABASE["tv_shows"][base_title]["seasons"]:
                                        CONTENT_DATABASE["tv_shows"][base_title]["seasons"].append(season_num)
                                else:
                                    logger.info(f"[DRY RUN] Would move '{season_path}' to '{dest_season_path}'")
                            except (OSError, shutil.Error) as e:
                                logger.error(f"Error moving '{season_folder_name}': {str(e)}")

                        # Handle special content folders
                        if special_folders:
                            logger.info(f"Processing {len(special_folders)} special content folders")

                            for special_path, special_name in special_folders:
                                # Format special folder name with show title
                                special_folder_name = f"{title_for_season} - {special_name}"
                                dest_special_path = os.path.join(tv_show_path, special_folder_name)

                                # Check if destination already exists
                                if os.path.exists(dest_special_path):
                                    logger.warning(f"Special folder already exists: {dest_special_path}")
                                    continue

                                # Move the special folder
                                try:
                                    if not DRY_RUN:
                                        logger.info(f"Moving special folder '{special_name}' to: {dest_special_path}")
                                        shutil.move(special_path, dest_special_path)
                                        logger.info(f"Successfully moved to: {dest_special_path}")
                                    else:
                                        logger.info(f"[DRY RUN] Would move '{special_path}' to '{dest_special_path}'")
                                except (OSError, shutil.Error) as e:
                                    logger.error(f"Error moving special folder '{special_name}': {str(e)}")

                        # Check if source folder is empty and remove if so
                        if not DRY_RUN and os.path.exists(folder_path) and not os.listdir(folder_path):
                            try:
                                os.rmdir(folder_path)
                                logger.info(f"Removed empty source folder: {folder_path}")
                            except OSError as e:
                                logger.error(f"Error removing empty folder '{folder_path}': {str(e)}")

                    except (OSError, shutil.Error) as e:
                        logger.error(f"Error processing TV seasons: {str(e)}")
                else:
                    # No seasons detected in the folder, check for special folders only
                    if special_folders:
                        logger.info(f"No seasons detected, but found {len(special_folders)} special folders.")
                        logger.info(f"Creating TV show directory and moving special folders")

                        try:
                            # Create TV show directory
                            if not DRY_RUN:
                                os.makedirs(tv_show_path, exist_ok=True)
                                logger.info(f"Created TV show directory: {tv_show_path}")
                            else:
                                logger.info(f"[DRY RUN] Would create TV show directory: {tv_show_path}")

                            # Move each special folder
                            for special_path, special_name in special_folders:
                                special_folder_name = f"{title_for_season} - {special_name}"
                                dest_special_path = os.path.join(tv_show_path, special_folder_name)

                                if os.path.exists(dest_special_path):
                                    logger.warning(f"Special folder already exists: {dest_special_path}")
                                    continue

                                if not DRY_RUN:
                                    logger.info(f"Moving special folder '{special_name}' to: {dest_special_path}")
                                    shutil.move(special_path, dest_special_path)
                                    logger.info(f"Successfully moved to: {dest_special_path}")
                                else:
                                    logger.info(f"[DRY RUN] Would move '{special_path}' to '{dest_special_path}'")

                            # Check if source folder is empty and remove if so
                            if not DRY_RUN and os.path.exists(folder_path) and not os.listdir(folder_path):
                                try:
                                    os.rmdir(folder_path)
                                    logger.info(f"Removed empty source folder: {folder_path}")
                                except OSError as e:
                                    logger.error(f"Error removing empty folder '{folder_path}': {str(e)}")
                        except (OSError, shutil.Error) as e:
                            logger.error(f"Error processing special folders: {str(e)}")
                    else:
                        # No seasons and no special folders, treat as a complete series
                        logger.info(f"No seasons or special folders detected. Treating as complete series.")
                        dest_path = os.path.join(TV_DIR, formatted_title)

                        # Handle existing destination
                        if os.path.exists(dest_path):
                            logger.warning(f"Found existing destination: {dest_path}")
                            counter = 1
                            while True:
                                new_dest_path = f"{dest_path} ({counter})"
                                if not os.path.exists(new_dest_path):
                                    dest_path = new_dest_path
                                    logger.info(f"Using alternative destination: {dest_path}")
                                    break
                                counter += 1

                        # Move the directory
                        logger.info(f"Moving entire folder to: {dest_path}")
                        try:
                            if not DRY_RUN:
                                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                                shutil.move(folder_path, dest_path)
                                logger.info(f"Successfully moved to: {dest_path}")

                                # Update database
                                base_title = re.sub(r'\s*\(\d{4}\)\s*', '', formatted_title).lower()
                                if base_title not in CONTENT_DATABASE["tv_shows"]:
                                    CONTENT_DATABASE["tv_shows"][base_title] = {
                                        "title": formatted_title,
                                        "year": re.search(r'\((\d{4})\)', formatted_title).group(1) if re.search(
                                            r'\(\d{4}\)', formatted_title) else "",
                                        "path": dest_path,
                                        "seasons": ["complete"]
                                    }
                            else:
                                logger.info(f"[DRY RUN] Would move '{folder_path}' to '{dest_path}'")
                        except (OSError, shutil.Error) as e:
                            logger.error(f"Error moving '{folder_name}': {str(e)}")

    elif content_type == "collection":
        # Handle movie collection
        if not HANDLE_COLLECTIONS:
            logger.info("Collection handling is disabled. Treating as a regular movie.")
            # Fallback to treating as a regular movie
            dest_path = os.path.join(MOVIES_DIR, formatted_title)
        else:
            # Move to collections directory
            dest_path = os.path.join(COLLECTIONS_DIR, formatted_title)

            # Check if destination exists
            if os.path.exists(dest_path):
                logger.warning(f"Collection already exists: {dest_path}")
                counter = 1
                while True:
                    new_dest_path = f"{dest_path} ({counter})"
                    if not os.path.exists(new_dest_path):
                        dest_path = new_dest_path
                        logger.info(f"Using alternative destination: {dest_path}")
                        break
                    counter += 1

            # Move the collection
            logger.info(f"Moving collection to: {dest_path}")
            try:
                if not DRY_RUN:
                    os.makedirs(COLLECTIONS_DIR, exist_ok=True)
                    shutil.move(folder_path, dest_path)
                    logger.info(f"Successfully moved to: {dest_path}")

                    # Organize the internal structure of the collection
                    organize_collection_content(dest_path)

                    # Add to database
                    base_title = re.sub(r'\s*\(\d{4}(?:-\d{4})?\)\s*', '', formatted_title).lower()
                    CONTENT_DATABASE["collections"][base_title] = {
                        "title": formatted_title,
                        "path": dest_path,
                        "year_range": re.search(r'\((\d{4}-\d{4})\)', formatted_title).group(1) if re.search(
                            r'\(\d{4}-\d{4}\)', formatted_title) else
                        re.search(r'\((\d{4})\)', formatted_title).group(1) if re.search(r'\(\d{4}\)',
                                                                                         formatted_title) else ""
                    }
                else:
                    logger.info(f"[DRY RUN] Would move '{folder_path}' to '{dest_path}'")
                    logger.info(f"[DRY RUN] Would organize the internal structure of the collection")
            except (OSError, shutil.Error) as e:
                logger.error(f"Error moving collection '{folder_name}': {str(e)}")

    else:
        logger.warning(f"Unknown content type for '{folder_name}', skipping.")


def scan_source_directory(recursive=RECURSIVE_SCAN, max_depth=MAX_SCAN_DEPTH):
    """Scan the source directory for new folders to process, with optional recursive scanning"""
    logger.info(f"Scanning source directory: {MAIN_DIR}")

    if recursive:
        logger.info(f"Recursive scanning enabled (max depth: {max_depth})")

    def _scan_directory(dir_path, current_depth=1):
        """Helper function for recursive scanning"""
        try:
            items = os.listdir(dir_path)

            if not items:
                logger.info(f"No items found in directory: {dir_path}")
                return

            # First pass - collect all folders to process
            folders_to_process = []

            for item in items:
                item_path = os.path.join(dir_path, item)

                # Only process directories
                if os.path.isdir(item_path):
                    # Skip if this is a Windows System Volume Information folder
                    if item == "System Volume Information":
                        continue

                    folders_to_process.append(item_path)

            # Process collected folders
            for folder_path in folders_to_process:
                try:
                    process_folder(folder_path)
                except Exception as e:
                    logger.error(f"Unhandled error processing folder '{folder_path}': {str(e)}")
                    # Continue with next folder rather than aborting completely

                # Add a small delay to prevent overloading the API
                time.sleep(1)

            # If recursive scanning is enabled and we haven't reached max depth,
            # scan subdirectories that might still exist after processing
            if recursive and current_depth < max_depth:
                # Re-check directory as it might have changed during processing
                if os.path.exists(dir_path):
                    for item in os.listdir(dir_path):
                        item_path = os.path.join(dir_path, item)
                        if os.path.isdir(item_path) and item != "System Volume Information":
                            _scan_directory(item_path, current_depth + 1)

        except OSError as e:
            logger.error(f"Error scanning directory '{dir_path}': {str(e)}")

    # Start the recursive scan from the main directory
    _scan_directory(MAIN_DIR)


def load_database():
    """Load database from file if it exists"""
    if os.path.exists(DB_FILE_PATH):
        try:
            with open(DB_FILE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Update global database
                for category in ["movies", "tv_shows", "collections"]:
                    if category in data:
                        CONTENT_DATABASE[category].update(data[category])
            logger.info(f"Loaded database from {DB_FILE_PATH}")
            return True
        except Exception as e:
            logger.error(f"Error loading database: {str(e)}")
    return False


def save_database():
    """Save database to file"""
    try:
        with open(DB_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(CONTENT_DATABASE, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved database to {DB_FILE_PATH}")
        return True
    except Exception as e:
        logger.error(f"Error saving database: {str(e)}")
        return False


def test_api_search():
    """Test the enhanced API search functionality"""
    test_queries = [
        "Clarkson's Farm",
        "Clarkson's Farm (2021)",
        "The Grand Tour",
        "Pacific Rim",
        "Blade Runner 2049",
        "Star Wars Episode IV",
        "The Matrix",
        "Top Gun",
        "Stranger Things",
        # Add tests for edge cases
        "Lord of the Rings Trilogy",
        "Star Wars Complete Saga 1977-2019",
        "Breaking Bad S01-S05 Complete",
        "Doctor Who Season 0 Specials",
        "Avatar (2009) Extended Collector's Edition",
        "Fast and Furious Collection 1-9",
        "Deadpool 1 & 2",
        "Dragon Ball Z Season 1 Namek Saga"
    ]

    logger.info("Testing Enhanced API Search...")

    # Create a session for all tests
    session = create_session()

    for query in test_queries:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Testing search for: {query}")
        logger.info(f"{'=' * 60}")

        # Extract year if present
        year_match = re.search(r'\((\d{4})\)', query)
        year = year_match.group(1) if year_match else None

        # Clean the query
        cleaned_query = clean_title_for_search(query)
        logger.info(f"Cleaned query: '{cleaned_query}'")

        # Run search
        result = search_media(cleaned_query, year, session)

        if result and "description" in result and result["description"]:
            logger.info(f"Found {len(result['description'])} results:")
            for idx, item in enumerate(result["description"][:5]):  # Show up to top 5 results
                title = item.get("#TITLE", "Unknown")
                year = item.get("#YEAR", "")
                media_type = item.get("#TYPE", "unknown")
                imdb_id = item.get("#IMDB_ID", "")
                source = item.get("#SOURCE_QUERY", "")
                logger.info(f"  Result #{idx + 1}: {title} ({year}) - Type: {media_type}")
                logger.info(f"             ID: {imdb_id}, Source Query: '{source}'")

            # Test content type determination
            logger.info("\nDetermining content type:")
            content_type, formatted_title = determine_content_type(result, query)
            logger.info(f"FINAL RESULT: {query} => {content_type.upper()} - '{formatted_title}'")
            logger.info(
                f"Would move to: {'TV' if content_type == 'tv' else 'COLLECTION' if content_type == 'collection' else 'Movies'} directory")
        else:
            logger.info("No results found")

        logger.info("-" * 60)


def test_ollama_api():
    """Test function to check if Ollama is working with sample titles including edge cases"""
    test_titles = [
        "The Matrix 1999",
        "Avatar 2009",
        "Star Wars A New Hope 1977",
        "The Mandalorian (2019) Season 2",
        "Clarkson's Farm (2021)",
        "Clarkson's Farm Season 2 (2023)",
        "The Grand Tour (2016)",
        "Top Gun (1986)",
        # Edge cases
        "Lord of the Rings Trilogy Extended Edition",
        "Star Wars The Complete Saga 1977-2019",
        "Breaking Bad S01-S05 Complete 1080p",
        "Doctor Who Specials Season 0",
        "Avatar (2009) Extended Collector's Edition",
        "Fast and Furious Collection 1-9",
        "Deadpool 1 & 2 Collection",
        "Dragon Ball Z Season 1 Namek Saga",
        "Jujutsu Kaisen Season 1 Cour 2 (2021)",
        "The Office US Complete Series S01-S09"
    ]

    logger.info("Testing Ollama API with sample titles...")
    logger.info(f"Using model: {OLLAMA_MODEL}")

    for title in test_titles:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Testing: {title}")
        logger.info(f"{'=' * 60}")

        try:
            result = clean_folder_name(title)
            logger.info(f"Cleaned title: {result}")

            # Also test the IMDb search
            logger.info("\nTesting IMDb search:")
            imdb_data = search_imdb(result, title)

            if imdb_data and "results" in imdb_data and imdb_data["results"]:
                for idx, item in enumerate(imdb_data["results"]):
                    logger.info(f"  Result #{idx + 1}: {item['title']} ({item['year']}) - Type: {item['type']}")
            else:
                logger.info("  No IMDb results found")

            # Finally test content type detection
            logger.info("\nTesting content type detection:")
            content_type, formatted_title = determine_content_type(imdb_data, title)
            logger.info(f"FINAL RESULT: {title} => {content_type.upper()} - '{formatted_title}'")
            logger.info(
                f"Would move to: {'TV' if content_type == 'tv' else 'COLLECTION' if content_type == 'collection' else 'Movies'} directory")
        except Exception as e:
            logger.error(f"Error processing test title '{title}': {str(e)}")

        logger.info("-" * 60)


def list_models():
    """Display available models and their descriptions"""
    logger.info("\nAvailable Ollama Models:")
    logger.info("-" * 50)
    for model, description in AVAILABLE_OLLAMA_MODELS.items():
        logger.info(f"- {model:<15} : {description}")
    logger.info("-" * 50)


def parse_arguments():
    """Parse command line arguments with enhanced options for edge cases"""
    parser = argparse.ArgumentParser(description='Organize media files into Movies and TV Show directories')

    # Basic options
    parser.add_argument('--dry-run', action='store_true',
                        help='Simulate organization without actually moving files')
    parser.add_argument('--skip-llm', action='store_true',
                        help='Skip LLM calls when direct title extraction is possible (faster)')
    parser.add_argument('--test-llm', action='store_true',
                        help='Run a test of the LLM functionality')
    parser.add_argument('--test-api', action='store_true',
                        help='Run a test of the API search functionality')

    # Model options
    parser.add_argument('--model', choices=list(AVAILABLE_OLLAMA_MODELS.keys()),
                        default=OLLAMA_MODEL,
                        help='Select the Ollama model to use')
    parser.add_argument('--no-shutdown', action='store_true',
                        help='Keep Ollama running after script completes')

    # API options
    parser.add_argument('--enable-tmdb', action='store_true',
                        help='Enable TMDb API as a fallback for content detection')
    parser.add_argument('--tmdb-key', type=str,
                        help='Your TMDb API key (required if --enable-tmdb is used)')

    # Directory options
    parser.add_argument('--source-dir', type=str,
                        help='Source directory to scan (overrides default)')
    parser.add_argument('--movies-dir', type=str,
                        help='Destination directory for movies (overrides default)')
    parser.add_argument('--tv-dir', type=str,
                        help='Destination directory for TV shows (overrides default)')
    parser.add_argument('--collections-dir', type=str,
                        help='Destination directory for movie collections (overrides default)')

    # Content handling options
    parser.add_argument('--recursive', action='store_true',
                        help='Enable recursive scanning of subdirectories')
    parser.add_argument('--max-depth', type=int, default=MAX_SCAN_DEPTH,
                        help='Maximum depth for recursive scanning')
    parser.add_argument('--disable-collections', action='store_true',
                        help='Disable special handling for movie collections')

    # Storage and Debug options
    parser.add_argument('--db-file', type=str, default=DB_FILE_PATH,
                        help='Path to database file for persistent storage')
    parser.add_argument('--show-details', action='store_true',
                        help='Show detailed information about content detection')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--log-file', type=str,
                        help='Path to log file (default: media_organizer.log)')

    # Special media handling
    parser.add_argument('--anime-mode', action='store_true',
                        help='Enable enhanced handling for anime content')
    parser.add_argument('--force-tv', nargs='+', default=[],
                        help='Force specific titles to be recognized as TV shows')
    parser.add_argument('--force-movie', nargs='+', default=[],
                        help='Force specific titles to be recognized as movies')

    return parser.parse_args()


def main():
    """Main function with enhanced error handling and configuration"""
    logger.info("Starting enhanced media organizer script...")

    # Parse command line arguments
    args = parse_arguments()

    # Update global variables based on command line arguments
    global ALWAYS_USE_LLM, OLLAMA_MODEL, TMDB_ENABLED, TMDB_API_KEY, DRY_RUN
    global RECURSIVE_SCAN, MAX_SCAN_DEPTH, HANDLE_COLLECTIONS
    global MAIN_DIR, MOVIES_DIR, TV_DIR, COLLECTIONS_DIR, DB_FILE_PATH

    # Configure directories
    if args.source_dir:
        MAIN_DIR = args.source_dir
    if args.movies_dir:
        MOVIES_DIR = args.movies_dir
    if args.tv_dir:
        TV_DIR = args.tv_dir
    if args.collections_dir:
        COLLECTIONS_DIR = args.collections_dir

    # Configure scanning options
    RECURSIVE_SCAN = args.recursive
    MAX_SCAN_DEPTH = args.max_depth
    HANDLE_COLLECTIONS = not args.disable_collections

    # Configure database
    if args.db_file:
        DB_FILE_PATH = args.db_file

    # Configure logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

    # Display available models
    list_models()

    # Set model from command line if provided
    if args.model:
        OLLAMA_MODEL = args.model
        logger.info(f"Using Ollama model: {OLLAMA_MODEL} - {AVAILABLE_OLLAMA_MODELS[OLLAMA_MODEL]}")

    # DRY_RUN setting
    if args.dry_run:
        DRY_RUN = True

    # ALWAYS_USE_LLM is True by default, and --skip-llm flag switches it off
    ALWAYS_USE_LLM = not args.skip_llm

    # Configure TMDb if requested
    if args.enable_tmdb:
        if args.tmdb_key:
            TMDB_API_KEY = args.tmdb_key
            TMDB_ENABLED = True
            logger.info("TMDb API enabled as a fallback for content detection")
        else:
            logger.error("Error: --enable-tmdb requires --tmdb-key to be provided")
            return

    # Update KNOWN_TV_SHOWS with any forced shows
    if args.force_tv:
        for show in args.force_tv:
            if show.lower() not in KNOWN_TV_SHOWS:
                KNOWN_TV_SHOWS.append(show.lower())
                logger.info(f"Added forced TV show: {show}")

    # Load existing database if available
    load_database()

    # Try to start Ollama if it's not running
    if not ensure_ollama_running():
        logger.warning("Warning: Unable to start Ollama. Title cleaning may fail.")

    # Display current mode
    if DRY_RUN:
        logger.info("Running in DRY RUN mode. No files will be moved.")
    else:
        logger.info("Running in ACTUAL MOVE mode. Files will be moved.")

    if ALWAYS_USE_LLM:
        logger.info("Using LLM for all title cleaning")
    else:
        logger.info("Using direct extraction when possible (faster)")

    # Test functions if requested
    if args.test_llm:
        test_ollama_api()
        return

    if args.test_api:
        test_api_search()
        return

    try:
        # Ensure destination directories exist
        for directory in [MOVIES_DIR, TV_DIR, COLLECTIONS_DIR]:
            if not DRY_RUN:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Ensured directory exists: {directory}")
            else:
                logger.info(f"[DRY RUN] Would create directory if it doesn't exist: {directory}")

        # Scan for new content
        scan_source_directory(recursive=RECURSIVE_SCAN, max_depth=MAX_SCAN_DEPTH)

        # Save the updated database
        if not DRY_RUN:
            save_database()

        if DRY_RUN:
            logger.info("Dry run complete! No files were actually moved.")
        else:
            logger.info("Processing complete! Files have been organized.")

    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        if args.verbose:
            logger.exception("Detailed traceback:")

    finally:
        # Shutdown Ollama to free up resources (unless --no-shutdown was specified)
        if not args.no_shutdown:
            shutdown_ollama()
        else:
            logger.info("Keeping Ollama running as requested (--no-shutdown)")


if __name__ == "__main__":
    main()