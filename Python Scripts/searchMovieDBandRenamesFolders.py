import os
import re
import shutil
import logging
import string

from rapidfuzz import fuzz  # pip install rapidfuzz
import requests

# ---------------------- Configuration ---------------------- #
MAIN_DIR = r"C:\Users\blue\Desktop\shared-folder\Complete"
PREFERRED_MEDIA_TYPE = "all"

IGNORE_PATTERNS = ["subs", "subtitles"]  # <--- Entire folders are skipped

# Words we want to strip out of queries to the API, even if they're in the folder name
API_IGNORE_WORDS = [
    "trailer", "sample", "dummy", "garbage",
    "1080p", "720p", "480p", "WEBRip", "WEB", "Complete",
    "H264", "H265", "10bit", "x264", "BluRay", "HDTV", "AAC", "DDP", "HEVC",
    "AMZN", "DL", "x265", "t3nzin", "BluRay", "5.1", "YTS.MX", "YTS"
]

MOVIES_DIR = r"C:\Users\blue\Desktop\shared-folder\Movies"
TV_DIR = r"C:\Users\blue\Desktop\shared-folder\TV"

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

# ---------------------- Helper Functions ---------------------- #
def should_ignore(folder_name):
    """
    Return True if folder_name matches any ignore pattern (entire folder is skipped).
    """
    return any(pattern in folder_name.lower() for pattern in IGNORE_PATTERNS)

# ---------------------- API SEARCH FUNCTIONS ---------------------- #
def search_movie(query):
    """
    Searches the Free Movie DB API at "https://imdb.iamidiotareyoutoo.com/search" with the given query.
    Returns parsed JSON result or None on error.
    """
    base_url = "https://imdb.iamidiotareyoutoo.com/search"
    params = {'q': query}  # Keep consistent with your API's expected param key.

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
    except Exception as e:
        logging.error(f"Network error for query '{query}': {e}")
        return None

    data = response.text
    logging.debug(f"Raw API response for '{query}':\n{data}\n")

    try:
        result = response.json()
        return result
    except Exception as e:
        logging.error(f"Error parsing JSON for query '{query}': {e}")
        return None


def generate_left_to_right_queries(query, candidate_date, media_type):
    """
    Generates a list of query variants by incrementally building
    substrings of the cleaned 'query' from left to right,
    optionally appending 'candidate_date' and media_type ("movie" or "tv") to each substring.

    Duplicate variants are removed while preserving order.
    """
    words = query.split()
    raw_variants = []

    # Build incremental queries
    for i in range(1, len(words) + 1):
        sub_query = ' '.join(words[:i])
        raw_variants.append(sub_query)

        # If there's a candidate date, try appending it
        if candidate_date:
            raw_variants.append(f"{sub_query} {candidate_date}")

        # If the media_type is specifically "movie" or "tv", also append that
        if media_type.lower() in ["movie", "tv"]:
            raw_variants.append(f"{sub_query} {media_type.lower()}")
            if candidate_date:
                raw_variants.append(f"{sub_query} {candidate_date} {media_type.lower()}")

    # Remove duplicates while preserving order
    seen = set()
    unique_variants = []
    for variant in raw_variants:
        if variant not in seen:
            seen.add(variant)
            unique_variants.append(variant)

    return unique_variants


def search_movie_variants(query, candidate_date, media_type="all"):
    """
    If 'media_type' is "movie" or "tv", generate variants just for that type.
    If 'media_type' is "all", generate both sets (movie and tv) and merge them.
    Then, call search_movie for each variant, collecting any results found.
    """
    if media_type.lower() in ["movie", "tv"]:
        variants = generate_left_to_right_queries(query, candidate_date, media_type)
    else:
        movie_variants = generate_left_to_right_queries(query, candidate_date, "movie")
        tv_variants = generate_left_to_right_queries(query, candidate_date, "tv")
        # Merge them, removing duplicates (dict.fromkeys keeps first occurrence)
        variants = list(dict.fromkeys(movie_variants + tv_variants))

    all_results = []
    for variant in variants:
        logging.info(f"Trying variant: '{variant}'")
        result = search_movie(variant)
        if result and "description" in result and result["description"]:
            all_results.extend(result["description"])

    return {"description": all_results} if all_results else None

# ---------------------- Title & Date Extraction ---------------------- #
def extract_title_and_date_from_folder(folder_name):
    """
    1) Removes resolution markers, SxxExx, 'Season N', etc.
    2) Strips out punctuation around each token, then drops any token in API_IGNORE_WORDS.
    3) Returns (cleaned_title, candidate_date).
    """
    base = os.path.splitext(folder_name)[0]

    # Identify a 4-digit year
    date_match = re.search(r'\b(19\d{2}|20\d{2})\b', base)
    candidate_date = date_match.group(0) if date_match else None
    if candidate_date in {"720", "1080", "2060"}:
        candidate_date = None

    # If there's something like 'Title Season 2', take only the part before 'Season'
    season_match = re.search(r'^(.*?)\s*[Ss]eason', base)
    if season_match:
        title_candidate = season_match.group(1)
    elif candidate_date:
        # Chop off everything from that year onward, if found
        title_candidate = base[:date_match.start()]
    else:
        title_candidate = base

    # Replace underscores/dots/dashes with spaces
    title_candidate = re.sub(r'[._\-]+', ' ', title_candidate)
    # Remove leftover patterns like S01E02, S01, Season 1, Episode 2, etc.
    title_candidate = re.sub(r'\b[Ss]\d{1,2}[Ee]\d{1,2}\b', '', title_candidate)
    title_candidate = re.sub(r'\b[Ss]\d{1,2}\b', '', title_candidate, flags=re.IGNORECASE)
    title_candidate = re.sub(r'\bSeason\s*\d+\b', '', title_candidate, flags=re.IGNORECASE)
    title_candidate = re.sub(r'\bEpisode\s*\d+\b', '', title_candidate, flags=re.IGNORECASE)
    # Remove typical resolution/quality markers
    title_candidate = re.sub(
        r'\b(1080p|720p|480p|WEBRip|WEB|Complete|H264|H265|10bit|x264|BluRay|HDTV|AAC|DDP|HEVC)\b',
        '',
        title_candidate,
        flags=re.IGNORECASE
    )

    # Split the result into tokens
    words = title_candidate.split()

    # Strip punctuation from each token's start/end, then remove if in API_IGNORE_WORDS
    filtered_words = []
    for w in words:
        # strip leading/trailing punctuation
        w_stripped = w.strip(string.punctuation)
        if w_stripped.lower() not in (x.lower() for x in API_IGNORE_WORDS):
            filtered_words.append(w_stripped)

    cleaned_title = " ".join(filtered_words).strip()

    # Remove extra spaces again
    cleaned_title = re.sub(r'\s+', ' ', cleaned_title).strip()

    return cleaned_title, candidate_date

def choose_best_match(results, query, candidate_date=None):
    """
    Chooses the best match from results using fuzzy matching.
    - If there's a dash, also test the part before the dash.
    - Apply a penalty if the candidate is too long.
    - Apply a bonus if candidate_date matches the result's #YEAR field.
    """
    best_ratio = -1
    best_candidate = None

    for candidate in results:
        candidate_title = candidate.get("#TITLE", "")

        full_score = fuzz.token_set_ratio(query.lower(), candidate_title.lower()) / 100.0

        # If there's a dash, also consider the left side
        if " - " in candidate_title:
            left_part = candidate_title.split(" - ")[0]
            left_score = fuzz.token_set_ratio(query.lower(), left_part.lower()) / 100.0
        else:
            left_score = full_score

        base_ratio = max(full_score, left_score)

        # If the candidate has many more words than the query, apply a penalty
        query_word_count = len(query.split())
        candidate_word_count = len(candidate_title.split())
        if candidate_word_count > query_word_count + 2:
            penalty = (candidate_word_count - (query_word_count + 2)) * 0.05
            base_ratio = max(0, base_ratio - penalty)

        # If there's a folder_year, see if the candidate's #YEAR matches it
        if candidate_date:
            candidate_year = str(candidate.get("#YEAR", ""))
            if candidate_year == candidate_date:
                ratio = base_ratio + 0.2  # small bonus
            else:
                ratio = base_ratio * 0.5  # penalty
        else:
            ratio = base_ratio

        if ratio > best_ratio:
            best_ratio = ratio
            best_candidate = candidate

    if best_candidate:
        return best_candidate
    else:
        # If empty results or no good match, fallback
        return results[0] if results else None

# ---------------------- Folder Processing ---------------------- #
def process_series_folder(folder_path, media_type="all"):
    """
    1) Clean & parse the folder name for a possible year.
    2) Search the API with left-to-right queries (movie+tv if 'all').
    3) If a good match is found, rename the folder.
    Returns (new_folder_path, final_series_title, final_api_media_type).
    """
    folder_name = os.path.basename(folder_path)
    logging.info(f"\nProcessing series folder: '{folder_name}'")

    extracted_title, candidate_date = extract_title_and_date_from_folder(folder_name)
    logging.info(f"Extracted title: '{extracted_title}' from folder: '{folder_name}'")
    if candidate_date:
        logging.info(f"Detected candidate date: '{candidate_date}'")

    result = search_movie_variants(extracted_title, candidate_date, media_type)
    if result and result.get("description"):
        best_match = choose_best_match(result["description"], extracted_title, candidate_date)
        if not best_match:
            logging.warning(f"No best match found for '{extracted_title}'. Keeping original name.")
            return folder_path, folder_name, None

        new_name = best_match.get("#TITLE", extracted_title)
        year = best_match.get("#YEAR", "")
        if year:
            new_name = f"{new_name} ({year})"

        # Remove illegal filename characters
        new_name = re.sub(r'[\\/*?:"<>|]', '', new_name)

        if folder_name != new_name:
            new_folder_path = os.path.join(os.path.dirname(folder_path), new_name)
            logging.info(f"Renaming series folder '{folder_name}' to '{new_name}'")
            try:
                os.rename(folder_path, new_folder_path)
                folder_path = new_folder_path
            except Exception as e:
                logging.error(f"Error renaming folder '{folder_name}': {e}")
        else:
            logging.info(f"Series folder '{folder_name}' is already correctly formatted.")

        series_title = new_name
        api_media_type = best_match.get("#TYPE", None)
        if api_media_type:
            api_media_type = api_media_type.lower()
    else:
        logging.warning(f"No results found for '{extracted_title}'. Keeping original folder name.")
        series_title = folder_name
        api_media_type = None

    return folder_path, series_title, api_media_type


def extract_season_number(folder_name):
    """
    Detect 'Season 2', 'S02', etc. and return that integer. None if not found.
    """
    pattern1 = re.compile(r"Season[\s._-]*(\d+)", re.IGNORECASE)
    pattern2 = re.compile(r"\bS[\s._-]?(\d{1,2})\b", re.IGNORECASE)

    match1 = pattern1.search(folder_name)
    if match1:
        return int(match1.group(1))

    match2 = pattern2.search(folder_name)
    if match2:
        return int(match2.group(1))

    return None


def process_season_folder(folder_path, series_title):
    """
    Renames subfolder 'S02' -> 'Series Title Season 2'
    """
    folder_name = os.path.basename(folder_path)
    logging.info(f"  Processing season folder: '{folder_name}'")

    season_number = extract_season_number(folder_name)
    if season_number is None:
        logging.info(f"    No season info found in '{folder_name}', skipping rename.")
        return

    new_name = f"{series_title} Season {season_number}"
    new_folder_path = os.path.join(os.path.dirname(folder_path), new_name)

    if folder_name != new_name:
        try:
            os.rename(folder_path, new_folder_path)
            logging.info(f"    Renamed season folder '{folder_name}' to '{new_name}'")
        except Exception as e:
            logging.error(f"    Error renaming season folder '{folder_name}': {e}")
    else:
        logging.info(f"    Season folder '{folder_name}' is already correctly formatted.")


def determine_media_type(series_folder_path):
    """
    If the folder name or subfolders mention 'Season', assume TV; otherwise, assume movie.
    """
    folder_name = os.path.basename(series_folder_path)
    if re.search(r'\s*[Ss]eason', folder_name):
        return "tv"

    try:
        for child in os.listdir(series_folder_path):
            child_path = os.path.join(series_folder_path, child)
            if os.path.isdir(child_path) and extract_season_number(child):
                return "tv"
    except Exception as e:
        logging.error(f"Error determining media type for '{series_folder_path}': {e}")

    return "movie"


def move_series_folder(folder_path, media_type):
    """
    Moves the folder to either the TV or Movies directory, based on media_type.
    """
    destination_dir = TV_DIR if media_type == "tv" else MOVIES_DIR

    if not os.path.exists(destination_dir):
        try:
            os.makedirs(destination_dir)
            logging.info(f"Created destination directory: {destination_dir}")
        except Exception as e:
            logging.error(f"Error creating destination directory '{destination_dir}': {e}")
            return folder_path

    folder_basename = os.path.basename(folder_path)
    new_location = os.path.join(destination_dir, folder_basename)
    try:
        shutil.move(folder_path, new_location)
        logging.info(f"Moved folder '{folder_basename}' to '{destination_dir}'.")
        return new_location
    except Exception as e:
        logging.error(f"Error moving folder '{folder_basename}' to '{destination_dir}': {e}")
        return folder_path


def process_main_directory(main_dir, media_type="all"):
    """
    1) Iterate all subdirs in main_dir (skipping IGNORE_PATTERNS).
    2) process_series_folder(...) to rename based on best match from the API.
    3) process_season_folder(...) on child subfolders.
    4) Heuristically finalize media type, move the entire folder.
    """
    logging.info(f"Starting processing of main directory: {main_dir}\n")

    for entry in list(os.listdir(main_dir)):
        entry_path = os.path.join(main_dir, entry)
        if os.path.isdir(entry_path):
            if should_ignore(entry):
                logging.info(f"Skipping folder (ignore pattern matched): '{entry}'")
                continue

            new_series_path, series_title, api_media_type = process_series_folder(entry_path, media_type)

            # Process season subfolders
            try:
                for child in os.listdir(new_series_path):
                    child_path = os.path.join(new_series_path, child)
                    if os.path.isdir(child_path) and not should_ignore(child):
                        process_season_folder(child_path, series_title)
            except Exception as e:
                logging.error(f"Error processing child folders in '{series_title}': {e}")

            # Decide final media type
            if api_media_type:
                if api_media_type in ["tv", "series"]:
                    determined_type = "tv"
                elif api_media_type == "movie":
                    determined_type = "movie"
                else:
                    determined_type = determine_media_type(new_series_path)
            else:
                determined_type = determine_media_type(new_series_path)

            logging.info(f"Determined media type for '{series_title}': {determined_type.upper()}")
            new_series_path = move_series_folder(new_series_path, determined_type)

    logging.info("\nProcessing complete.")


# ---------------------- Entry Point ---------------------- #
if __name__ == "__main__":
    process_main_directory(MAIN_DIR, media_type=PREFERRED_MEDIA_TYPE)
