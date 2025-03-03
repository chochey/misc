import os
import re
import logging
import string
import time
import datetime
import configparser
import shutil
from rapidfuzz import fuzz
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import datefinder
from functools import lru_cache
import argparse
import multiprocessing

# ---------------------- DIRECTORY SETTINGS ---------------------- #
MAIN_DIR = r"I:\share-downloads\complete"  # Source directory to scan
MOVIES_DIR = r"I:\Media3\Movies"  # Destination for movie files
TV_DIR = r"I:\Media3\TV"  # Destination for TV show files


# ---------------------- Configuration Class ---------------------- #
class Config:
    def __init__(self, config_file=None):
        # Default values
        self.MAIN_DIR = MAIN_DIR
        self.PREFERRED_MEDIA_TYPE = "all"
        self.IGNORE_PATTERNS = ["subs", "subtitles", "sample"]
        self.API_IGNORE_WORDS = [
            "trailer", "sample", "dummy", "garbage", "1080p", "720p", "480p",
            "WEBRip", "WEB", "Complete", "H264", "H265", "10bit", "x264",
            "BluRay", "HDTV", "AAC", "DDP", "HEVC", "AMZN", "DL", "x265",
            "BluRay", "5.1", "YTS", "uncut", "unrated", "french", "eng",
            "rarbg", "yify", "remastered", "remux", "HDR", "UHD", "DTS", "MultiSub", "iTA",
            "MIRCrew", "AsPiDe",
        ]
        self.MOVIES_DIR = MOVIES_DIR
        self.TV_DIR = TV_DIR
        self.MIN_FUZZ_RATIO = 0.6
        self.MAX_WORKERS = min(5, multiprocessing.cpu_count())
        self.VIDEO_EXTENSIONS = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm']
        self.API_URL = "https://imdb.iamidiotareyoutoo.com/search"
        self.MAX_RETRIES = 3
        self.RETRY_DELAY = 5
        self.RETRY_STATUS_CODES = [429, 500, 502, 503, 504]
        self.BACKOFF_FACTOR = 0.3
        self.API_TIMEOUT = 10
        self.INTERACTIVE_MODE = False
        self.LOG_LEVEL = "INFO"
        self.PREFER_SUBTITLES = False  # Default to False - prefer titles without subtitles
        self.MAX_SEARCH_QUERIES = 10  # Maximum number of search queries to try
        self.SEARCH_DELAY = 0.5  # Delay between searches to avoid rate limiting (seconds)

        # Load from config file if provided
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)

    def load_config(self, config_file):
        """Load configuration from file."""
        config = configparser.ConfigParser()
        config.read(config_file)

        if 'Paths' in config:
            paths = config['Paths']
            self.MAIN_DIR = os.path.expanduser(paths.get('MAIN_DIR', self.MAIN_DIR))
            self.MOVIES_DIR = os.path.expanduser(paths.get('MOVIES_DIR', self.MOVIES_DIR))
            self.TV_DIR = os.path.expanduser(paths.get('TV_DIR', self.TV_DIR))

        if 'Settings' in config:
            settings = config['Settings']
            self.PREFERRED_MEDIA_TYPE = settings.get('PREFERRED_MEDIA_TYPE', self.PREFERRED_MEDIA_TYPE)
            self.MIN_FUZZ_RATIO = float(settings.get('MIN_FUZZ_RATIO', str(self.MIN_FUZZ_RATIO)))
            self.MAX_WORKERS = int(settings.get('MAX_WORKERS', str(self.MAX_WORKERS)))
            self.INTERACTIVE_MODE = settings.getboolean('INTERACTIVE_MODE', self.INTERACTIVE_MODE)
            self.LOG_LEVEL = settings.get('LOG_LEVEL', self.LOG_LEVEL)
            self.PREFER_SUBTITLES = settings.getboolean('PREFER_SUBTITLES', self.PREFER_SUBTITLES)
            self.MAX_SEARCH_QUERIES = int(settings.get('MAX_SEARCH_QUERIES', str(self.MAX_SEARCH_QUERIES)))
            self.SEARCH_DELAY = float(settings.get('SEARCH_DELAY', str(self.SEARCH_DELAY)))

            # Parse extensions
            video_exts = settings.get('VIDEO_EXTENSIONS', None)
            if video_exts:
                self.VIDEO_EXTENSIONS = [ext.strip() for ext in video_exts.split(',')]

            # Parse ignored patterns
            ignore_patterns = settings.get('IGNORE_PATTERNS', None)
            if ignore_patterns:
                self.IGNORE_PATTERNS = [pattern.strip() for pattern in ignore_patterns.split(',')]

        if 'API' in config:
            api = config['API']
            self.API_URL = api.get('API_URL', self.API_URL)
            self.MAX_RETRIES = int(api.get('MAX_RETRIES', str(self.MAX_RETRIES)))
            self.RETRY_DELAY = int(api.get('RETRY_DELAY', str(self.RETRY_DELAY)))
            self.API_TIMEOUT = int(api.get('API_TIMEOUT', str(self.API_TIMEOUT)))

    def save_config(self, config_file='media_organizer.ini'):
        """Save current configuration to file."""
        config = configparser.ConfigParser()

        config['Paths'] = {
            'MAIN_DIR': self.MAIN_DIR,
            'MOVIES_DIR': self.MOVIES_DIR,
            'TV_DIR': self.TV_DIR
        }

        config['Settings'] = {
            'PREFERRED_MEDIA_TYPE': self.PREFERRED_MEDIA_TYPE,
            'MIN_FUZZ_RATIO': str(self.MIN_FUZZ_RATIO),
            'MAX_WORKERS': str(self.MAX_WORKERS),
            'VIDEO_EXTENSIONS': ','.join(self.VIDEO_EXTENSIONS),
            'IGNORE_PATTERNS': ','.join(self.IGNORE_PATTERNS),
            'INTERACTIVE_MODE': str(self.INTERACTIVE_MODE),
            'LOG_LEVEL': self.LOG_LEVEL,
            'PREFER_SUBTITLES': str(self.PREFER_SUBTITLES),
            'MAX_SEARCH_QUERIES': str(self.MAX_SEARCH_QUERIES),
            'SEARCH_DELAY': str(self.SEARCH_DELAY)
        }

        config['API'] = {
            'API_URL': self.API_URL,
            'MAX_RETRIES': str(self.MAX_RETRIES),
            'RETRY_DELAY': str(self.RETRY_DELAY),
            'API_TIMEOUT': str(self.API_TIMEOUT)
        }

        with open(config_file, 'w') as f:
            config.write(f)
        return config_file


# ---------------------- Logging Setup ---------------------- #
def setup_logging(log_level="INFO"):
    """Configure logging based on selected level."""
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR
    }
    level = level_map.get(log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler()]
    )


# ---------------------- HTTP Session Setup ---------------------- #
def create_session(config):
    """Create and configure HTTP session with retry handling."""
    session = requests.Session()
    retry_strategy = Retry(
        total=config.MAX_RETRIES,
        backoff_factor=config.BACKOFF_FACTOR,
        status_forcelist=config.RETRY_STATUS_CODES
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session


# ---------------------- Helper Functions ---------------------- #
def should_ignore(folder_name, config):
    """Check if a folder should be ignored based on patterns."""
    return any(pattern.lower() in folder_name.lower() for pattern in config.IGNORE_PATTERNS)


def safe_rename(src, dst, dry_run=False):
    """Rename a file or folder, handling collisions by appending a number."""
    if not os.path.exists(src):
        logging.error(f"Source path does not exist: {src}")
        return src

    if dry_run:
        logging.info(f"[DRY RUN] Would rename '{src}' to '{dst}'")
        return src

    if os.path.exists(dst):
        base, ext = os.path.splitext(dst)
        counter = 1
        new_dst = f"{base} ({counter}){ext}"
        while os.path.exists(new_dst):
            counter += 1
            new_dst = f"{base} ({counter}){ext}"
        logging.warning(f"Collision detected. Renaming '{dst}' -> '{new_dst}'")
        os.rename(src, new_dst)
        return new_dst
    else:
        os.rename(src, dst)
        return dst


def safe_move_with_retry(src, dst, config, dry_run=False):
    """Move a file with retry logic."""
    if dry_run:
        logging.info(f"[DRY RUN] Would move '{src}' to '{dst}'")
        return True, src

    retries = 0
    while retries < config.MAX_RETRIES:
        try:
            # Make sure destination directory exists
            os.makedirs(os.path.dirname(dst), exist_ok=True)

            if os.path.exists(dst):
                base, ext = os.path.splitext(dst)
                counter = 1
                new_dst = f"{base} ({counter}){ext}"
                while os.path.exists(new_dst):
                    counter += 1
                    new_dst = f"{base} ({counter}){ext}"
                dst = new_dst
                logging.warning(f"Destination exists, using '{dst}' instead")

            shutil.move(src, dst)
            logging.info(f"Moved '{src}' to '{dst}'")
            return True, dst
        except Exception as e:
            logging.warning(f"Error moving file: {e}. Retry {retries + 1}/{config.MAX_RETRIES}")
            time.sleep(config.RETRY_DELAY)
            retries += 1

    logging.error(f"Failed to move '{src}' to '{dst}' after {config.MAX_RETRIES} attempts")
    return False, src


def merge_into_existing_folder(src_folder, dst_folder, config, dry_run=False):
    """Merge contents of src_folder into dst_folder."""
    if not os.path.exists(dst_folder):
        if not dry_run:
            os.makedirs(dst_folder)
        logging.info(f"Created destination directory: {dst_folder}")

    for item in os.listdir(src_folder):
        src_path = os.path.join(src_folder, item)
        dst_path = os.path.join(dst_folder, item)

        if os.path.isfile(src_path):
            success, _ = safe_move_with_retry(src_path, dst_path, config, dry_run)
            if not success:
                logging.warning(f"Failed to move '{src_path}' to '{dst_folder}'")
        elif os.path.isdir(src_path):
            dst_subdir = os.path.join(dst_folder, item)
            if os.path.exists(dst_subdir):
                # If subdirectory exists, merge recursively
                merge_into_existing_folder(src_path, dst_subdir, config, dry_run)
            else:
                # Move entire subdirectory
                if dry_run:
                    logging.info(f"[DRY RUN] Would move directory '{src_path}' to '{dst_subdir}'")
                else:
                    try:
                        shutil.move(src_path, dst_subdir)
                    except Exception as e:
                        logging.error(f"Failed to move directory '{src_path}': {e}")

    # Remove empty source folder
    if not dry_run and os.path.exists(src_folder) and not os.listdir(src_folder):
        shutil.rmtree(src_folder)


def get_unique_folder_name(base_path, folder_name):
    """Generate a unique folder name by appending a number if necessary."""
    new_folder_path = os.path.join(base_path, folder_name)
    counter = 1
    while os.path.exists(new_folder_path):
        new_folder_path = os.path.join(base_path, f"{folder_name} ({counter})")
        counter += 1
    return new_folder_path


def validate_year(year_str):
    """Validate a year string based on reasonable media year range."""
    try:
        year = int(year_str)
        current_year = datetime.datetime.now().year
        return 1890 <= year <= (current_year + 3)
    except ValueError:
        return False


# ---------------------- TV/Movie Detection Functions ---------------------- #
def extract_tv_show_info(file_name):
    """Extract show name, season, and episode from a file name."""
    patterns = [
        r"(?P<show>.+?)[._ -]+[Ss](?P<season>\d{1,2})[Ee](?P<episode>\d{1,2})",
        r"(?P<show>.+?)[._ -]+Season[._ -]+(?P<season>\d+)[._ -]+Episode[._ -]+(?P<episode>\d+)",
        r"(?P<show>.+?)[._ -]+(?P<season>\d{1,2})x(?P<episode>\d{1,2})",
        r"(?P<show>.+?)[._ -]+[Ee]pisode[._ -]+(?P<episode>\d+)",
        r"(?P<show>.+?)[._ -]+[Ss]pecial[._ -]+(?P<episode>\d{1,2})"
    ]

    for pattern in patterns:
        match = re.search(pattern, file_name, re.IGNORECASE)
        if match:
            try:
                show = match.group("show").replace('.', ' ').strip()

                # Handle special cases
                if "season" in match.groupdict():
                    season = int(match.group("season"))
                elif "special" in file_name.lower():
                    season = 0  # Use season 0 for specials
                else:
                    season = 1  # Default to season 1 for patterns with no season

                episode = int(match.group("episode"))
                return show, season, episode
            except Exception:
                continue

    return None, None, None


def extract_series_and_season(folder_name, config):
    """Extract series title and season number from folder name."""
    base = os.path.splitext(folder_name)[0]
    season_patterns = [
        r"(.*?)\s*(?:Season\s*(\d+)|[Ss](\d{1,2}))",
        r"(.*?)\s*\(Season\s*(\d+)\)",
    ]
    for pattern in season_patterns:
        match = re.search(pattern, base, re.IGNORECASE)
        if match:
            series_part = match.group(1).strip()
            try:
                season_number = int(match.group(2) or match.group(3))
                if series_part:
                    series_title, candidate_date, _ = extract_title_and_date_from_folder(series_part, config)
                    return series_title, candidate_date, season_number
            except Exception:
                continue
    return None, None, None


def extract_title_and_date_from_folder(folder_name, config):
    """Extract title and date from a folder name."""
    # Get base name without extension
    base = os.path.splitext(folder_name)[0]

    # Replace typical separators with spaces
    base = re.sub(r'[._\-]+', ' ', base)

    # Find year in common patterns
    year_patterns = [
        r'\b(19\d{2}|20\d{2})\b',  # Standard year format
        r'\((\d{4})\)',  # Year in parentheses
        r'\[(\d{4})\]',  # Year in brackets
    ]

    candidate_date = None
    for pattern in year_patterns:
        year_match = re.search(pattern, base)
        if year_match:
            year = year_match.group(1)
            if validate_year(year):
                candidate_date = year
                # Remove the year from the title
                base = re.sub(pattern, '', base).strip()
                break

    # If no year found, try datefinder
    if not candidate_date:
        try:
            matches = list(datefinder.find_dates(base))
            for match in matches:
                year = str(match.year)
                if validate_year(year):
                    candidate_date = year
                    base = re.sub(r'\b' + year + r'\b', '', base).strip()
                    break
        except Exception:
            pass

    # Clean title - Remove quality indicators, encodings, etc.
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
    if hasattr(config, 'API_IGNORE_WORDS') and config.API_IGNORE_WORDS:
        ignore_pattern = '|'.join(
            r'\b' + re.escape(word) + r'\b'
            for word in config.API_IGNORE_WORDS
        )
        base = re.sub(ignore_pattern, '', base, flags=re.IGNORECASE)

    # Final cleanup
    base = re.sub(r'\s+', ' ', base).strip()
    base = base.strip(string.punctuation + ' ')
    base = re.sub(r'\(\s*\)', '', base)
    base = re.sub(r'\[\s*\]', '', base)

    cleaned_title = base.strip()

    # Strip enclosing quotes if present
    if (cleaned_title.startswith('"') and cleaned_title.endswith('"')) or \
            (cleaned_title.startswith("'") and cleaned_title.endswith("'")):
        cleaned_title = cleaned_title[1:-1]

    return cleaned_title, candidate_date, None


def extract_season_number(folder_name):
    """Extract season number from folder name."""
    patterns = [
        r"\b[Ss]eason\s*(\d+)\b",
        r"\b[Ss](\d{1,2})\b"
    ]
    for pattern in patterns:
        match = re.search(pattern, folder_name, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                continue
    return None


def has_episode_files(folder_path, config):
    """Check if a folder contains episode files."""
    patterns = [r"\b\d{1,2}[xE]\d{1,2}\b", r"E\d{1,2}\b", r"Episode\s*\d+"]
    try:
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path) and os.path.splitext(file_name.lower())[1] in config.VIDEO_EXTENSIONS:
                for pattern in patterns:
                    if re.search(pattern, file_name, re.IGNORECASE):
                        return True
    except Exception:
        pass
    return False


def determine_media_type(folder_path, api_media_type, has_seasons, config):
    """Determine if a folder is a movie or TV show."""
    folder_name = os.path.basename(folder_path)

    # Trust API result if available
    if api_media_type in ["tv", "movie"]:
        return api_media_type

    # Check for obvious season indicators
    if has_seasons or re.search(r'\s*[Ss]eason', folder_name, re.IGNORECASE):
        return "tv"

    # Check for episode files
    if has_episode_files(folder_path, config):
        return "tv"

    # Check number of video files - movies typically have 1-2 files
    try:
        video_files = [f for f in os.listdir(folder_path)
                       if os.path.isfile(os.path.join(folder_path, f))
                       and os.path.splitext(f.lower())[1] in config.VIDEO_EXTENSIONS]
        if len(video_files) > 2:  # More than 2 video files likely a TV show
            return "tv"
    except Exception:
        pass

    # Default to movie if uncertain
    return "movie"


# ---------------------- API Search Functions ---------------------- #
@lru_cache(maxsize=100)
def search_movie(query, api_url, session, timeout=10):
    """Search for a movie or TV show using the API."""
    params = {'q': query}
    try:
        response = session.get(api_url, params=params, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict) or "description" not in data:
            logging.error(f"Unexpected API response format for '{query}'")
            return None
        return data
    except Exception as e:
        logging.error(f"API error for query '{query}': {e}")
        return None


def search_media(title, year=None, config=None, session=None):
    """
    Enhanced search for media with multi-stage fallback and sophisticated ranking.

    Args:
        title: Cleaned title string
        year: Optional year
        config: Configuration object
        session: HTTP session

    Returns:
        Best match results or None
    """
    if not title or not config or not session:
        return None

    # Track all results across multiple query attempts
    all_results = []
    queries_tried = set()

    # STAGE 1: Generate and try primary queries
    primary_queries = generate_primary_queries(title, year)

    # Check for exact match first
    exact_match_result = try_exact_match_search(primary_queries, title, year, config, session, queries_tried)
    if exact_match_result:
        return exact_match_result

    # Get results from primary queries
    primary_results = execute_search_queries(primary_queries, config, session, queries_tried)
    all_results.extend(primary_results)

    # STAGE 2: Try title variations if we don't have good results yet
    if not is_good_result_set(all_results, title, year):
        # Generate title variations for problem titles
        title_variations = generate_title_variations(title)

        for variation in title_variations:
            var_queries = generate_primary_queries(variation, year)
            var_results = execute_search_queries(var_queries, config, session, queries_tried)
            all_results.extend(var_results)

            # If we get good results from a variation, we can stop
            if is_good_result_set(var_results, variation, year):
                logging.info(f"Found good results using title variation: '{variation}'")
                break

    # STAGE 3: Try alternative queries if still needed
    if not is_good_result_set(all_results, title, year) and len(all_results) < 5:
        alt_queries = generate_alternative_queries(title, year)
        alt_results = execute_search_queries(alt_queries, config, session, queries_tried)
        all_results.extend(alt_results)

    # Final processing: deduplicate, score, and rank results
    if not all_results:
        logging.warning(f"No results found for '{title}'")
        return None

    ranked_results = process_search_results(all_results, title, year, config)

    return {"description": ranked_results} if ranked_results else None


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

    if year:
        # Try with movie type + year
        queries.append(f"{title} movie {year}")
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


def try_exact_match_search(queries, original_title, year, config, session, queries_tried):
    """Try to find an exact title match from primary queries."""
    original_title_lower = original_title.lower().strip()

    for query in queries:
        if query in queries_tried or len(query.strip()) < 2:
            continue

        logging.info(f"Trying exact match query: '{query}'")
        queries_tried.add(query)

        result = search_movie(query, config.API_URL, session, config.API_TIMEOUT)

        if result and "description" in result and result["description"]:
            for item in result["description"]:
                item_title = item.get("#TITLE", "").lower().strip()

                # Check for exact title match
                if item_title == original_title_lower:
                    if year and str(item.get("#YEAR", "")) == str(year):
                        logging.info(f"Found exact match with year: '{item.get('#TITLE')}' ({year})")
                        return {"description": [item]}
                    else:
                        logging.info(f"Found exact title match: '{item.get('#TITLE')}'")
                        return {"description": [item]}

                # Check for exact main title match (before subtitle)
                if ":" in item_title:
                    main_part = item_title.split(":")[0].strip()
                    if main_part == original_title_lower:
                        # For problematic titles, prefer match without subtitle
                        problematic_titles = ["pacific rim uprising", "blade runner", "thor", "star wars"]
                        if original_title_lower in problematic_titles:
                            # Create modified item with just the main title
                            modified_item = item.copy()
                            modified_item["#TITLE"] = item.get("#TITLE", "").split(":")[0].strip()
                            logging.info(
                                f"Found main title match for problematic title: '{modified_item.get('#TITLE')}'")
                            return {"description": [modified_item]}
                        else:
                            logging.info(f"Found main title match: '{item.get('#TITLE')}'")
                            return {"description": [item]}

    return None


def execute_search_queries(queries, config, session, queries_tried):
    """Execute a list of search queries and collect all results."""
    results = []

    for query in queries:
        if query in queries_tried or len(query.strip()) < 2:
            continue

        logging.info(f"Trying query: '{query}'")
        queries_tried.add(query)

        result = search_movie(query, config.API_URL, session, config.API_TIMEOUT)

        if result and "description" in result and result["description"]:
            for item in result["description"]:
                # Add the source query for later scoring/analysis
                item_copy = item.copy()
                item_copy["#SOURCE_QUERY"] = query
                results.append(item_copy)

        # Respect rate limits - add delay between queries
        if config.SEARCH_DELAY > 0:
            time.sleep(config.SEARCH_DELAY)

        # Check if we've reached the maximum number of queries
        if len(queries_tried) >= config.MAX_SEARCH_QUERIES:
            logging.info(f"Reached maximum search queries limit ({config.MAX_SEARCH_QUERIES})")
            break

    return results


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


def process_search_results(results, title, year, config):
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
            logging.info(f"Best match: '{item.get('#TITLE')}' with score {score:.2f}")
        else:
            logging.info(f"Alternative match {i}: '{item.get('#TITLE')}' with score {score:.2f}")

    # Check for problematic titles - modify the top result if needed
    top_score, top_item = scored_results[0] if scored_results else (0, None)

    if top_item:
        problematic_titles = ["pacific rim uprising", "blade runner", "thor", "star wars"]
        item_title = top_item.get("#TITLE", "").lower()

        if title.lower() in problematic_titles and ":" in item_title and not config.PREFER_SUBTITLES:
            # Clean the title by removing the subtitle
            modified_item = top_item.copy()
            main_title = top_item.get("#TITLE").split(":")[0].strip()
            modified_item["#TITLE"] = main_title
            logging.info(f"Using clean title without subtitle for problematic match: '{main_title}'")

            # Replace the top item with the modified version
            scored_results[0] = (top_score, modified_item)

    # Apply minimum score threshold (adjust based on config)
    normalized_threshold = config.MIN_FUZZ_RATIO * 100
    filtered_results = [(score, item) for score, item in scored_results if score >= normalized_threshold]

    if not filtered_results:
        logging.warning(f"No results met minimum threshold of {normalized_threshold:.2f}")
        if scored_results:
            # Return the best match we have even if it's below threshold
            logging.info(f"Using best available match: '{scored_results[0][1].get('#TITLE')}'")
            return [item for _, item in scored_results]
        return []

    # Return items in ranked order
    return [item for _, item in filtered_results]


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

    # Bonus for being the expected media type
    media_type = candidate.get("#TYPE", "").lower()
    if media_type == "movie":
        content_bonus += 5

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


def choose_best_match(results, query, candidate_date=None, min_ratio=0.6, interactive=False):
    """Choose the best match from search results."""
    if not results:
        return None

    # Score each result
    scored_results = [(enhanced_match_quality(candidate, query, candidate_date), candidate)
                      for candidate in results]

    # Sort by score (highest first)
    scored_results.sort(reverse=True, key=lambda x: x[0])

    # Interactive selection if enabled
    if interactive and len(scored_results) > 1:
        print("\nPotential matches found. Please select the correct one:")
        for i, (score, candidate) in enumerate(scored_results[:min(10, len(scored_results))]):
            title = candidate.get("#TITLE", "Unknown")
            year = candidate.get("#YEAR", "Unknown")
            media_type = candidate.get("#TYPE", "Unknown")
            print(f"{i + 1}. {title} ({year}) - {media_type.upper()} [Score: {score / 100:.2f}]")

        try:
            choice = int(input("\nEnter number (0 to skip): "))
            if choice == 0:
                return None
            if 1 <= choice <= len(scored_results):
                return scored_results[choice - 1][1]
        except (ValueError, IndexError):
            print("Invalid selection, using best automatic match")

    # Check if best match is good enough
    if scored_results:
        best_score, best_candidate = scored_results[0]
        normalized_score = best_score / 100.0

        if normalized_score < min_ratio:
            logging.warning(f"Best match score ({normalized_score:.2f}) below threshold ({min_ratio})")
            return None

        logging.info(f"Best match: '{best_candidate.get('#TITLE')}' with score {normalized_score:.2f}")
        return best_candidate

    return None


# ---------------------- Folder Processing ---------------------- #
def process_series_folder(folder_path, config, session, dry_run=False):
    """Process a folder, renaming based on API results with improved search and matching."""
    folder_name = os.path.basename(folder_path)

    # Check if it's a standalone season folder
    series_title, candidate_date, season_number = extract_series_and_season(folder_name, config)
    if series_title and season_number:
        logging.info(f"Detected season folder: '{folder_name}'")

        result = search_media(series_title, candidate_date, config, session)
        best_match = None
        if result and result.get("description"):
            # Use the first result as the best match - the enhanced search already ranks results
            best_match = result["description"][0] if result["description"] else None

        if best_match:
            series_title = best_match["#TITLE"]
            year = best_match["#YEAR"] or candidate_date or ""
            if year:
                series_title = f"{series_title} ({year})"
            series_title = re.sub(r'[\\/*?:"<>|]', '', series_title)
            season_folder_name = f"{series_title} Season {season_number}"
            new_folder_path = os.path.join(os.path.dirname(folder_path), season_folder_name)
            final_path = safe_rename(folder_path, new_folder_path, dry_run)

            # Create parent series folder and move season inside
            destination = os.path.join(config.TV_DIR, series_title)
            if not os.path.exists(destination) and not dry_run:
                os.makedirs(destination, exist_ok=True)

            season_destination = os.path.join(destination, os.path.basename(final_path))
            final_location = safe_rename(final_path, season_destination, dry_run)
            return final_location, series_title, "tv", season_number

        return folder_path, folder_name, None, season_number

    # Process as main series folder
    extracted_title, candidate_date, _ = extract_title_and_date_from_folder(folder_name, config)
    logging.info(f"Extracted title: '{extracted_title}'")

    # Special handling for problematic titles
    problematic_titles = ["pacific rim uprising", "blade runner", "thor", "star wars"]
    if extracted_title.lower() in problematic_titles or any(
            title in extracted_title.lower() for title in problematic_titles):
        logging.info(f"Detected potentially problematic title: '{extracted_title}'")

    # Use the enhanced search logic
    result = search_media(extracted_title, candidate_date, config, session)

    if result and result.get("description"):
        # The search_media function now returns already processed and ranked results
        # The first item should be the best match
        best_match = result["description"][0] if result["description"] else None

        if not best_match:
            logging.warning(f"No match for '{extracted_title}'. Keeping original name.")
            return folder_path, folder_name, None, None

        new_name = best_match.get("#TITLE", extracted_title)
        year = best_match["#YEAR"] or candidate_date or ""
        if year:
            new_name = f"{new_name} ({year})"
        new_name = re.sub(r'[\\/*?:"<>|]', '', new_name)

        if folder_name != new_name:
            new_folder_path = os.path.join(os.path.dirname(folder_path), new_name)
            logging.info(f"Renaming '{folder_name}' to '{new_name}'")
            final_path = safe_rename(folder_path, new_folder_path, dry_run)
        else:
            final_path = folder_path

        series_title = new_name
        api_media_type = best_match.get("#TYPE", "movie")
        logging.info(f"API media type: '{api_media_type}'")
    else:
        logging.warning(f"No results for '{extracted_title}'. Keeping original name.")
        series_title = folder_name
        api_media_type = None

    return final_path, series_title, api_media_type, None


def process_season_folder(folder_path, series_title, dry_run=False):
    """Process and rename a season folder."""
    folder_name = os.path.basename(folder_path)
    season_number = extract_season_number(folder_name)
    if not season_number:
        return False

    new_name = f"{series_title} Season {season_number}"
    new_folder_path = os.path.join(os.path.dirname(folder_path), new_name)

    if folder_name != new_name:
        final_path = safe_rename(folder_path, new_folder_path, dry_run)
        logging.info(f"Renamed season folder to '{os.path.basename(final_path)}'")

    return True


def move_series_folder(folder_path, media_type, series_title, config, dry_run=False):
    """Move a folder to the appropriate media directory."""
    destination_dir = config.TV_DIR if media_type == "tv" else config.MOVIES_DIR

    # Create destination directory if doesn't exist
    if not os.path.exists(destination_dir):
        if not dry_run:
            os.makedirs(destination_dir, exist_ok=True)

    existing_folder = os.path.join(destination_dir, series_title)

    if os.path.exists(existing_folder):
        logging.info(f"Merging into existing folder '{existing_folder}'")
        merge_into_existing_folder(folder_path, existing_folder, config, dry_run)
        return existing_folder
    else:
        new_location = os.path.join(destination_dir, os.path.basename(folder_path))
        final_location = safe_rename(folder_path, new_location, dry_run)
        logging.info(f"Moved to '{final_location}'")
        return final_location


def process_loose_video_files(main_dir, config, dry_run=False):
    """Process loose video files in the main directory."""
    try:
        loose_files = [
            f for f in os.listdir(main_dir)
            if os.path.isfile(os.path.join(main_dir, f))
               and os.path.splitext(f.lower())[1] in config.VIDEO_EXTENSIONS
        ]
    except Exception as e:
        logging.error(f"Could not access directory '{main_dir}': {e}")
        return []

    if not loose_files:
        return []

    # Group files by show name for TV shows
    tv_show_groups = {}
    movie_files = []

    for file in loose_files:
        file_path = os.path.join(main_dir, file)
        show_name, season, episode = extract_tv_show_info(file)

        if show_name and season is not None and episode is not None:
            if show_name not in tv_show_groups:
                tv_show_groups[show_name] = []
            tv_show_groups[show_name].append((file_path, season, episode))
        else:
            movie_files.append(file_path)

    organized_folders = []

    # Process TV show groups
    for show_name, episodes in tv_show_groups.items():
        show_folder = get_unique_folder_name(main_dir, show_name)
        if not dry_run:
            os.makedirs(show_folder, exist_ok=True)

        # Organize episodes into season folders
        season_groups = {}
        for ep in episodes:
            file_path, season, episode = ep
            if season not in season_groups:
                season_groups[season] = []
            season_groups[season].append((file_path, episode))

        for season, eps in season_groups.items():
            season_folder = os.path.join(show_folder, f"Season {season}")
            if not dry_run:
                os.makedirs(season_folder, exist_ok=True)

            for file_path, episode in eps:
                new_file_path = os.path.join(season_folder, os.path.basename(file_path))
                safe_move_with_retry(file_path, new_file_path, config, dry_run)

        organized_folders.append(show_folder)

    # Process movie files
    for movie_file in movie_files:
        file_name = os.path.splitext(os.path.basename(movie_file))[0]
        cleaned_title, candidate_date, _ = extract_title_and_date_from_folder(file_name, config)

        if candidate_date:
            movie_title = f"{cleaned_title} ({candidate_date})"
        else:
            movie_title = cleaned_title

        movie_folder = get_unique_folder_name(main_dir, movie_title)

        if not dry_run:
            os.makedirs(movie_folder, exist_ok=True)
            new_file_path = os.path.join(movie_folder, os.path.basename(movie_file))
            safe_move_with_retry(movie_file, new_file_path, config, dry_run)
        else:
            logging.info(f"[DRY RUN] Would move '{movie_file}' to '{movie_folder}'")

        organized_folders.append(movie_folder)

    return organized_folders


def process_main_directory(main_dir, config, session, dry_run=False):
    """Process the main directory."""
    logging.info(f"Starting processing of {main_dir}")
    final_results = []

    # Verify directory exists
    if not os.path.isdir(main_dir):
        logging.error(f"Directory does not exist: {main_dir}")
        return []

    # Process loose video files
    process_loose_video_files(main_dir, config, dry_run)

    # Get all folders to process
    try:
        folders = [os.path.join(main_dir, entry) for entry in os.listdir(main_dir)
                   if os.path.isdir(os.path.join(main_dir, entry))
                   and not should_ignore(entry, config)]
    except Exception as e:
        logging.error(f"Could not list directory '{main_dir}': {e}")
        return []

    # Process each folder
    for folder_path in folders:
        try:
            new_series_path, series_title, api_media_type, season_number = process_series_folder(
                folder_path, config, session, dry_run
            )

            if season_number is not None:
                final_results.append({"title": series_title, "location": new_series_path})
            else:
                has_seasons = False
                try:
                    for child in os.listdir(new_series_path):
                        child_path = os.path.join(new_series_path, child)
                        if os.path.isdir(child_path) and not should_ignore(child, config):
                            if process_season_folder(child_path, series_title, dry_run):
                                has_seasons = True
                except Exception:
                    pass

                determined_type = determine_media_type(new_series_path, api_media_type, has_seasons, config)

                # If we found seasons but determined it's not TV, override to TV
                if has_seasons and determined_type != "tv":
                    determined_type = "tv"

                new_series_path = move_series_folder(
                    new_series_path,
                    determined_type,
                    series_title,
                    config,
                    dry_run
                )

                final_results.append({
                    "title": series_title,
                    "location": new_series_path,
                    "type": determined_type
                })
        except Exception as e:
            logging.error(f"Error processing folder '{folder_path}': {e}")

    logging.info("Processing complete.")
    for item in final_results:
        media_type = item.get("type", "unknown")
        logging.info(f"Title: {item['title']} ({media_type}), Location: {item['location']}")

    return final_results


# ---------------------- Command-Line Interface ---------------------- #
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Media folder organizer")
    parser.add_argument("--main-dir", help="Main directory to process")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--media-type", choices=["all", "movie", "tv"], help="Media type to prefer")
    parser.add_argument("--dry-run", action="store_true", help="Simulate actions without making changes")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive mode for ambiguous matches")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set logging level")
    parser.add_argument("--create-config", help="Create a default configuration file at the specified path")
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Create default config if requested
    if args.create_config:
        config = Config()
        config_file = config.save_config(args.create_config)
        print(f"Created default configuration file at: {config_file}")
        return

    # Load configuration
    config = Config(args.config)

    # Override config with command-line arguments
    if args.main_dir:
        if not os.path.isdir(args.main_dir):
            print(f"Error: {args.main_dir} is not a valid directory")
            return
        config.MAIN_DIR = os.path.abspath(args.main_dir)
    if args.media_type:
        config.PREFERRED_MEDIA_TYPE = args.media_type
    if args.interactive:
        config.INTERACTIVE_MODE = True
    if args.log_level:
        config.LOG_LEVEL = args.log_level

    # Set up logging
    setup_logging(config.LOG_LEVEL)

    # Create HTTP session
    session = create_session(config)

    # Process directory
    process_main_directory(
        config.MAIN_DIR,
        config,
        session,
        args.dry_run
    )


if __name__ == "__main__":
    main()