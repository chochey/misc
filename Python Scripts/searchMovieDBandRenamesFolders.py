import os
import re
import http.client
import urllib.parse
import json
import difflib

# ---------------------- Configuration ---------------------- #
# Set your main directory path (adjust as needed)
MAIN_DIR = r"C:\Users\blue\Desktop\shared-folder\Complete"
# Set your preferred media type ("movie", "tv", or "all")
PREFERRED_MEDIA_TYPE = "all"
# Folders containing these substrings (case‑insensitive) will be skipped.
IGNORE_PATTERNS = ["subs", "subtitles"]


# ---------------------- Helper Functions ---------------------- #
def should_ignore(folder_name):
    """
    Returns True if folder_name contains any ignore pattern.
    """
    lower_name = folder_name.lower()
    return any(pattern in lower_name for pattern in IGNORE_PATTERNS)


# ---------------------- API SEARCH FUNCTIONS ---------------------- #
def search_movie(query):
    """
    Searches the Free Movie DB API for a title based on the query.
    Returns the parsed JSON result, or None if something goes wrong.
    """
    conn = http.client.HTTPSConnection("imdb.iamidiotareyoutoo.com")
    params = urllib.parse.urlencode({'q': query})
    path = f"/search?{params}"
    try:
        conn.request("GET", path)
        res = conn.getresponse()
    except Exception as e:
        print(f"Network error for query '{query}': {e}")
        conn.close()
        return None

    if res.status != 200:
        print(f"Error fetching data for query '{query}'. HTTP Status: {res.status}")
        conn.close()
        return None

    data = res.read().decode("utf-8")
    # Debug: print the raw API response
    print(f"Raw API response for '{query}':\n{data}\n")
    conn.close()
    try:
        result = json.loads(data)
        return result
    except Exception as e:
        print(f"Error parsing JSON for query '{query}': {e}")
        return None


def extract_title_and_date_from_folder(folder_name):
    """
    Extracts a cleaned title and candidate date from a folder name.

    Steps:
      1. Remove any file extension.
      2. Look for a candidate date (a 4-digit number between 1900 and 2099).
      3. If "Season" appears, assume it's a TV show and take only the portion before it.
      4. Otherwise, if a candidate date was found, use the part before the date.
      5. Replace common delimiters (dots, underscores, hyphens) with spaces.
      6. Remove abbreviated season/episode markers (like S03E01) and full words "Season" or "Episode" with numbers.
      7. Remove common resolution/encoding tags.
      8. Collapse extra whitespace.

    Returns a tuple: (cleaned_title, candidate_date)
    """
    # Remove any file extension.
    base = os.path.splitext(folder_name)[0]

    # Look for a candidate date (a 4-digit number between 1900 and 2099)
    date_match = re.search(r'\b(19\d{2}|20\d{2})\b', base)
    candidate_date = date_match.group(0) if date_match else None
    if candidate_date in {"720", "1080", "2060"}:
        candidate_date = None

    # If "Season" appears in the folder name, assume it's a TV show.
    season_match = re.search(r'^(.*?)\s*[Ss]eason', base)
    if season_match:
        title_candidate = season_match.group(1)
    elif candidate_date:
        title_candidate = base[:date_match.start()]
    else:
        title_candidate = base

    # Replace common delimiters with a space.
    title_candidate = re.sub(r'[._\-]+', ' ', title_candidate)
    # Remove abbreviated season/episode markers (e.g., S03E01)
    title_candidate = re.sub(r'\b[Ss]\d{1,2}[Ee]\d{1,2}\b', '', title_candidate)
    # Remove full words "Season" or "Episode" with numbers.
    title_candidate = re.sub(r'\bSeason\s*\d+\b', '', title_candidate, flags=re.IGNORECASE)
    title_candidate = re.sub(r'\bEpisode\s*\d+\b', '', title_candidate, flags=re.IGNORECASE)
    # Remove common resolution/encoding tags.
    title_candidate = re.sub(r'\b(1080p|720p|480p|WEB|H264|x264|BluRay|HDTV|AAC|DDP|HEVC)\b', '', title_candidate,
                             flags=re.IGNORECASE)
    # Collapse multiple spaces and trim.
    cleaned_title = re.sub(r'\s+', ' ', title_candidate).strip()
    return cleaned_title, candidate_date


def search_movie_variants(query, candidate_date, media_type="all"):
    """
    Tries several query variants based on the cleaned query and candidate date.
    Aggregates and returns all results (under the "description" key) from all variants.
    """
    if media_type.lower() == "movie":
        if candidate_date:
            variants = [f"{query} {candidate_date} movie", f"{query} {candidate_date} film",
                        f"{query} movie", f"{query} film", query]
        else:
            variants = [f"{query} movie", f"{query} film", query]
    elif media_type.lower() == "tv":
        if candidate_date:
            variants = [f"{query} {candidate_date} tv", f"{query} {candidate_date} series",
                        f"{query} tv", f"{query} series", query]
        else:
            variants = [f"{query} tv", f"{query} series", query]
    else:  # media_type "all"
        if candidate_date:
            variants = [f"{query} {candidate_date}", f"{query} {candidate_date} movie", f"{query} {candidate_date} tv",
                        f"{query} movie", f"{query} film", f"{query} tv", f"{query} series", query]
        else:
            variants = [f"{query} movie", f"{query} film", f"{query} tv", f"{query} series", query]

    all_results = []
    for variant in variants:
        print(f"Trying variant: '{variant}'")
        result = search_movie(variant)
        if result and "description" in result and len(result["description"]) > 0:
            all_results.extend(result["description"])
    if all_results:
        return {"description": all_results}
    return None


# ---------------------- Updated Matching Function ---------------------- #
def choose_best_match(results, query, candidate_date=None):
    """
    Chooses the best match from the aggregated results using fuzzy matching.

    - Computes a base ratio comparing the query and candidate title.
    - If candidate_date is provided:
         - Adds a bonus (0.2) if the candidate's "#YEAR" matches candidate_date.
         - Otherwise, penalizes the candidate by multiplying the ratio by 0.5.
    - Returns the candidate with the highest adjusted score.
    """
    best_ratio = -1  # Start with a very low ratio.
    best_candidate = None
    for candidate in results:
        candidate_title = candidate.get("#TITLE", "")
        base_ratio = difflib.SequenceMatcher(None, query.lower(), candidate_title.lower()).ratio()
        if candidate_date:
            candidate_year = str(candidate.get("#YEAR", ""))
            if candidate_year == candidate_date:
                ratio = base_ratio + 0.2  # Bonus for a year match.
            else:
                ratio = base_ratio * 0.5  # Penalize if the year doesn't match.
        else:
            ratio = base_ratio
        if ratio > best_ratio:
            best_ratio = ratio
            best_candidate = candidate
    return best_candidate if best_candidate else results[0]


# ---------------------- Folder Processing Functions ---------------------- #
def process_series_folder(folder_path, media_type="all"):
    """
    Processes a top-level (series) folder:
      1. Extracts a cleaned title and candidate date from the folder name.
      2. Uses the API (via several query variants) to search for an official title.
      3. Chooses the best match and renames the folder to "Official Title (Year)".
      4. Returns the (possibly updated) folder path and the official series title.
    """
    folder_name = os.path.basename(folder_path)
    print(f"\nProcessing series folder: '{folder_name}'")

    # Extract a cleaned title and candidate date.
    extracted_title, candidate_date = extract_title_and_date_from_folder(folder_name)
    print(f"Extracted title: '{extracted_title}' from folder: '{folder_name}'")
    if candidate_date:
        print(f"Detected candidate date: '{candidate_date}'")

    # Search for API results using several query variants.
    result = search_movie_variants(extracted_title, candidate_date, media_type)
    if result and "description" in result and len(result["description"]) > 0:
        best_match = choose_best_match(result["description"], extracted_title, candidate_date)
        # Build new folder name using the API fields.
        new_name = best_match.get("#TITLE", extracted_title)
        year = best_match.get("#YEAR")
        if year:
            new_name = f"{new_name} ({year})"
        # Sanitize the new folder name for illegal filesystem characters.
        new_name = re.sub(r'[\\/*?:"<>|]', '', new_name)
        if folder_name != new_name:
            new_folder_path = os.path.join(os.path.dirname(folder_path), new_name)
            print(f"Renaming series folder '{folder_name}' to '{new_name}'")
            try:
                os.rename(folder_path, new_folder_path)
                folder_path = new_folder_path  # Update path after renaming.
            except Exception as e:
                print(f"Error renaming folder '{folder_name}': {e}")
        else:
            print(f"Series folder '{folder_name}' is already correctly formatted.")
        series_title = new_name
    else:
        print(f"No results found for '{extracted_title}'. Keeping original folder name.")
        series_title = folder_name
    return folder_path, series_title


def extract_season_number(folder_name):
    """
    Detects season information in a folder name using robust regular expressions.
    Supports patterns like "Season 1" and abbreviated forms like "S01".
    Returns the season number as an integer if found; otherwise, returns None.
    """
    # Pattern for full word: "Season 1", "Season-1", etc.
    pattern1 = re.compile(r"Season[\s._-]*(\d+)", re.IGNORECASE)
    # Pattern for abbreviated form: "S01", "S-01", etc.
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
    Processes a child (season) folder:
      1. Checks for season information in the folder name.
      2. If found, renames the folder to "Series Title Season X" (using the parent's series title).
    """
    folder_name = os.path.basename(folder_path)
    print(f"  Processing season folder: '{folder_name}'")
    season_number = extract_season_number(folder_name)
    if season_number is None:
        print(f"    No season info found in '{folder_name}', skipping rename.")
        return
    new_name = f"{series_title} Season {season_number}"
    new_folder_path = os.path.join(os.path.dirname(folder_path), new_name)
    if folder_name != new_name:
        try:
            os.rename(folder_path, new_folder_path)
            print(f"    Renamed season folder '{folder_name}' to '{new_name}'")
        except Exception as e:
            print(f"    Error renaming season folder '{folder_name}': {e}")
    else:
        print(f"    Season folder '{folder_name}' is already correctly formatted.")


def process_main_directory(main_dir, media_type="all"):
    """
    Processes the main directory:
      1. Iterates over each immediate subdirectory (series folder).
      2. For each series folder, processes it via the API search and renaming.
      3. Then iterates over each child folder inside the series folder and renames those
         if season information is detected.
    """
    print(f"Starting processing of main directory: {main_dir}\n")
    for entry in os.listdir(main_dir):
        entry_path = os.path.join(main_dir, entry)
        if os.path.isdir(entry_path):
            if should_ignore(entry):
                print(f"Skipping folder (ignore pattern matched): '{entry}'")
                continue
            # Process the top-level (series) folder.
            new_series_path, series_title = process_series_folder(entry_path, media_type)
            # Process each immediate child (season) folder in the series folder.
            try:
                for child in os.listdir(new_series_path):
                    child_path = os.path.join(new_series_path, child)
                    if os.path.isdir(child_path):
                        if should_ignore(child):
                            print(f"  Skipping folder (ignore pattern matched): '{child}'")
                            continue
                        process_season_folder(child_path, series_title)
            except Exception as e:
                print(f"Error processing child folders in '{series_title}': {e}")
    print("\nProcessing complete.")


# ---------------------- Entry Point ---------------------- #
if __name__ == "__main__":
    process_main_directory(MAIN_DIR, media_type=PREFERRED_MEDIA_TYPE)
