#!/usr/bin/env python3
"""Movie Renamer - Automatically rename and organize movie files."""

import argparse
import html
import json
import re
import shutil
import time
import logging
from difflib import SequenceMatcher
from pathlib import Path

import requests
from PTN import parse

import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Track consecutive OMDb 401 errors to detect rate limiting
_omdb_consecutive_401s = 0
_OMDB_RATE_LIMIT_THRESHOLD = 5  # Abort after this many consecutive 401s

# Cache OMDb "no match" results to avoid re-querying the same files in watch mode
# Key: (title, year), Value: timestamp of when the lookup failed
_omdb_no_match_cache: dict[tuple[str, str | None], float] = {}
_OMDB_CACHE_TTL = 3600  # Cache "no match" results for 1 hour


class OMDbRateLimitError(Exception):
    """Raised when OMDb API rate limit is detected."""
    pass


def _check_omdb_rate_limit(response: requests.Response) -> None:
    """Track consecutive 401 errors and raise if rate limited."""
    global _omdb_consecutive_401s
    if response.status_code == 401:
        _omdb_consecutive_401s += 1
        if _omdb_consecutive_401s >= _OMDB_RATE_LIMIT_THRESHOLD:
            raise OMDbRateLimitError(
                f"OMDb API rate limit reached ({_omdb_consecutive_401s} consecutive 401 errors). "
                f"The free tier allows 1,000 requests/day. Try again tomorrow."
            )
    else:
        _omdb_consecutive_401s = 0


def find_existing_movie_drive(movie_folder_name: str) -> Path | None:
    """Check all movie drives for an existing folder with this name."""
    for drive_dir in config.MOVIE_DEST_DIRS:
        candidate = Path(drive_dir) / movie_folder_name
        if candidate.exists():
            return Path(drive_dir)
    return None


def find_existing_show_drive(series_folder_name: str) -> Path | None:
    """Check all TV drives for an existing folder with this name."""
    for drive_dir in config.TV_DEST_DIRS:
        candidate = Path(drive_dir) / series_folder_name
        if candidate.exists():
            return Path(drive_dir)
    return None


def get_movie_dest(movie_folder_name: str) -> Path:
    """Get the destination drive for a movie, preferring drives that already have it."""
    existing = find_existing_movie_drive(movie_folder_name)
    if existing:
        return existing
    return Path(config.MOVIE_DEST_DIR)


def get_tv_dest(series_folder_name: str) -> Path:
    """Get the destination drive for a TV show, preferring drives that already have it."""
    existing = find_existing_show_drive(series_folder_name)
    if existing:
        return existing
    return Path(config.TV_DEST_DIR)


def parse_filename(filename: str) -> tuple[str | None, str | None]:
    """Extract title and year from a filename using PTN."""
    parsed = parse(filename)
    title = parsed.get("title")
    year = parsed.get("year")
    return title, str(year) if year else None


def clean_title(title: str) -> str:
    """Clean a title by removing common release group tags and normalizing."""
    release_tags = [
        r"\[TGx\]", r"\[YTS[^\]]*\]", r"\[RARBG\]", r"\[ettv\]", r"\[eztv\]",
        r"YTS\.MX", r"YTS\.LT", r"YTS\.AM", r"RARBG", r"TGx",
        r"\(TGx\)", r"\(RARBG\)", r"\(YTS[^\)]*\)",
    ]
    for tag in release_tags:
        title = re.sub(tag, "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def get_title_variations(title: str) -> list[str]:
    """Generate variations of a title to try searching.

    Variations are ordered from most specific to least specific.
    Aggressive strategies (word removal) are excluded to prevent false matches.
    """
    title = clean_title(title)
    variations = [title]

    # Try title before colon (e.g., "Wake Up Dead Man: A Knives Out Mystery" -> "Wake Up Dead Man")
    if ":" in title:
        before_colon = title.split(":")[0].strip()
        if before_colon and before_colon not in variations:
            variations.append(before_colon)

    # Try removing common subtitle patterns
    patterns = [
        r"\s+A\s+\w+\s+(Out\s+)?(Mystery|Story|Tale|Film|Movie).*$",
        r"\s*[-:]\s*A\s+\w+\s+(Mystery|Story|Tale|Film|Movie).*$",
        r"\s*[-:]\s*The\s+\w+.*$",
        r"\s*[-:]\s*(Horror|Action|Comedy|Drama|Thriller|Romance|Sci-Fi|SciFi|Adventure|Fantasy|Animation|Documentary|Crime|Western|Musical|War|History|Sport|Family|Biography)$",
    ]
    for pattern in patterns:
        cleaned = re.sub(pattern, "", title, flags=re.IGNORECASE).strip()
        if cleaned and cleaned != title and cleaned not in variations:
            variations.append(cleaned)

    # Try removing informal sequel numbers: "Title 2 - Subtitle" -> "Title Subtitle"
    # (e.g., "Transformers 2 - Revenge of The Fallen" -> "Transformers Revenge of The Fallen")
    no_seq = re.sub(r"\s+\d+\s*[-:]\s*", " ", title).strip()
    if no_seq and no_seq != title and no_seq not in variations:
        variations.append(no_seq)

    # Add "and" <-> "&" variations
    for var in variations.copy():
        if " and " in var.lower():
            swapped = re.sub(r"\band\b", "&", var, flags=re.IGNORECASE)
            if swapped not in variations:
                variations.append(swapped)
        if " & " in var:
            swapped = var.replace(" & ", " and ")
            if swapped not in variations:
                variations.append(swapped)

    # NOTE: Progressive word removal has been removed intentionally.
    # It caused false matches (e.g., "The Lord of the Rings" -> "The Lord").

    return variations


def is_good_match(parsed_title: str, parsed_year: str | None, omdb_info: dict) -> bool:
    """Check if the OMDb result is a confident match for what we parsed.

    Uses title similarity and year proximity to filter out bad matches
    like TV specials, talk show appearances, or wrong movies entirely.
    """
    omdb_title = omdb_info.get("title", "")
    omdb_year = omdb_info.get("year", "")
    omdb_type = omdb_info.get("type", "movie")

    # Reject non-movie types (episodes, series, etc.)
    if omdb_type and omdb_type.lower() != "movie":
        log.info(f"  Rejected: type is '{omdb_type}', not 'movie'")
        return False

    # Title similarity check
    title_similarity = SequenceMatcher(
        None,
        parsed_title.lower().strip(),
        omdb_title.lower().strip(),
    ).ratio()

    # Year proximity check (allow ±1 for regional release date differences)
    year_ok = True
    if parsed_year and omdb_year:
        try:
            # OMDb year can be a range like "2023-2024", take the first year
            omdb_year_int = int(omdb_year[:4])
            parsed_year_int = int(parsed_year[:4])
            year_ok = abs(parsed_year_int - omdb_year_int) <= 1
        except ValueError:
            year_ok = False

    # Strict: if title is very different, reject
    if title_similarity < 0.5:
        log.info(
            f"  Rejected: title similarity too low ({title_similarity:.2f}) "
            f"'{parsed_title}' vs '{omdb_title}'"
        )
        return False

    # If year doesn't match, require very high title similarity
    if not year_ok:
        if title_similarity < 0.85:
            log.info(
                f"  Rejected: year mismatch ({parsed_year} vs {omdb_year}) "
                f"and title similarity only {title_similarity:.2f}"
            )
            return False
        else:
            log.warning(
                f"  Warning: year mismatch ({parsed_year} vs {omdb_year}) "
                f"but title similarity high ({title_similarity:.2f}), accepting"
            )

    # Check for suspiciously long titles that might indicate a TV special
    # e.g., "Movie Name: Late Night with Host featuring Guest Stars"
    suspicious_patterns = [
        r"guest\s*star",
        r"late\s*(night|show)",
        r"talk\s*show",
        r"interview",
        r"behind\s*the\s*scenes",
        r"making\s*of",
        r"special\s*appearance",
        r"featuring",
        r"hosted\s*by",
        r"with\s+(jimmy|stephen|seth|jimmy|james|conan|trevor|john)",
    ]
    for pattern in suspicious_patterns:
        if re.search(pattern, omdb_title, re.IGNORECASE):
            log.info(f"  Rejected: suspicious title pattern in '{omdb_title}' (likely TV special)")
            return False

    log.info(f"  Match accepted: similarity={title_similarity:.2f}, year_ok={year_ok}")
    return True


def query_omdb_exact(title: str, year: str | None = None) -> dict | None:
    """Query OMDb API with exact title match (t= parameter).

    Raises OMDbRateLimitError if too many consecutive 401 errors are detected.
    """
    params = {
        "apikey": config.OMDB_API_KEY,
        "t": title,
        "type": "movie",
    }
    if year:
        params["y"] = year

    try:
        response = requests.get(config.OMDB_BASE_URL, params=params, timeout=10)
        _check_omdb_rate_limit(response)
        response.raise_for_status()
        data = response.json()

        if data.get("Response") == "True":
            return {
                "title": data.get("Title"),
                "year": data.get("Year"),
                "type": data.get("Type"),
            }
    except OMDbRateLimitError:
        raise
    except requests.RequestException as e:
        log.error(f"OMDb API error: {e}")

    return None


def query_omdb_search(title: str, year: str | None = None) -> dict | None:
    """Query OMDb API with search (s= parameter) and return best match.

    Raises OMDbRateLimitError if too many consecutive 401 errors are detected.
    """
    params = {
        "apikey": config.OMDB_API_KEY,
        "s": title,
        "type": "movie",
    }
    if year:
        params["y"] = year

    try:
        response = requests.get(config.OMDB_BASE_URL, params=params, timeout=10)
        _check_omdb_rate_limit(response)
        response.raise_for_status()
        data = response.json()

        if data.get("Response") == "True" and data.get("Search"):
            first_result = data["Search"][0]
            return {
                "title": first_result.get("Title"),
                "year": first_result.get("Year"),
                "type": first_result.get("Type"),
            }
    except OMDbRateLimitError:
        raise
    except requests.RequestException as e:
        log.error(f"OMDb API error: {e}")

    return None


def query_omdb(title: str, year: str | None = None) -> dict | None:
    """Query OMDb API, trying multiple strategies to find a match.

    Strategies are ordered from most reliable to least reliable.
    Each match is validated with is_good_match() before being accepted.
    Results are cached to avoid re-querying the same titles in watch mode.
    """
    cache_key = (title.lower().strip(), year)
    cached_time = _omdb_no_match_cache.get(cache_key)
    if cached_time and (time.time() - cached_time) < _OMDB_CACHE_TTL:
        return None

    variations = get_title_variations(title)

    # Strategy 1: Exact match with year for all variations (most reliable)
    for variant in variations:
        result = query_omdb_exact(variant, year)
        if result and is_good_match(title, year, result):
            log.info(f"  Strategy: exact + year, query: '{variant}'")
            return result
        # Small delay to avoid hammering the API
        time.sleep(0.15)

    # Strategy 2: Exact match WITHOUT year (year from filename might be wrong)
    if year:
        for variant in variations[:3]:  # Limit to top 3 variations
            result = query_omdb_exact(variant, None)
            if result and is_good_match(title, year, result):
                log.info(f"  Strategy: exact (no year), query: '{variant}'")
                return result
            time.sleep(0.15)

    # Strategy 3: Search endpoint with year (less reliable, limited attempts)
    for variant in variations[:2]:  # Only try first 2 variations
        result = query_omdb_search(variant, year)
        if result and is_good_match(title, year, result):
            log.info(f"  Strategy: search + year, query: '{variant}'")
            return result
        time.sleep(0.15)

    # Strategy 4: Search endpoint WITHOUT year (least reliable, very limited)
    if year:
        result = query_omdb_search(variations[0], None)  # Only try original title
        if result and is_good_match(title, year, result):
            log.info(f"  Strategy: search (no year), query: '{variations[0]}'")
            return result

    _omdb_no_match_cache[cache_key] = time.time()
    return None


def get_all_files(directory: Path, extensions: list[str]) -> list[Path]:
    """Recursively find all files with given extensions in a directory."""
    files = []
    for f in directory.rglob("*"):
        if f.is_file() and f.suffix.lower() in extensions:
            files.append(f)
    return files


def is_still_transferring(name: str) -> bool:
    """Check if a file/folder with the same name exists in the incomplete folder."""
    incomplete_dir = Path(config.INCOMPLETE_DIR)
    if not incomplete_dir.exists():
        return False

    for item in incomplete_dir.iterdir():
        if item.name.lower() == name.lower():
            return True
        if item.stem.lower() == name.lower():
            return True

    return False


def is_tv_show(parsed: dict) -> bool:
    """Check if parsed filename indicates a TV show (has season/episode info)."""
    return bool(parsed.get("season") or parsed.get("episode"))


def parse_tv_info(filename: str) -> dict | None:
    """Parse TV show info from filename. Returns None if not a TV show."""
    parsed = parse(filename)
    if not is_tv_show(parsed):
        return None

    episode = parsed.get("episode")
    # PTN returns a list for multi-episode files (e.g., S01E15E16) - use the first
    if isinstance(episode, list):
        episode = episode[0] if episode else None

    return {
        "title": parsed.get("title"),
        "season": parsed.get("season", 1),
        "episode": episode,
        "episode_title": parsed.get("episodeName"),
    }


def query_omdb_series(title: str, year: str | None = None) -> dict | None:
    """Query OMDb API for a TV series.

    Raises OMDbRateLimitError if too many consecutive 401 errors are detected.
    Results are cached to avoid re-querying the same titles in watch mode.
    """
    cache_key = (f"series:{title.lower().strip()}", year)
    cached_time = _omdb_no_match_cache.get(cache_key)
    if cached_time and (time.time() - cached_time) < _OMDB_CACHE_TTL:
        return None

    params = {
        "apikey": config.OMDB_API_KEY,
        "t": title,
        "type": "series",
    }
    if year:
        params["y"] = year

    try:
        response = requests.get(config.OMDB_BASE_URL, params=params, timeout=10)
        _check_omdb_rate_limit(response)
        response.raise_for_status()
        data = response.json()

        if data.get("Response") == "True":
            return {
                "title": data.get("Title"),
                "year": data.get("Year", "").split("–")[0],  # Get start year from "2008–2013"
                "type": data.get("Type"),
            }
    except OMDbRateLimitError:
        raise  # Let this propagate up
    except requests.RequestException as e:
        log.error(f"OMDb API error: {e}")

    _omdb_no_match_cache[cache_key] = time.time()
    return None


def query_omdb_episode(series_title: str, season: int, episode: int) -> dict | None:
    """Query OMDb API for episode info to get episode title.

    Raises OMDbRateLimitError if too many consecutive 401 errors are detected.
    """
    params = {
        "apikey": config.OMDB_API_KEY,
        "t": series_title,
        "Season": season,
        "Episode": episode,
    }

    try:
        response = requests.get(config.OMDB_BASE_URL, params=params, timeout=10)
        _check_omdb_rate_limit(response)
        response.raise_for_status()
        data = response.json()

        if data.get("Response") == "True":
            return {
                "episode_title": data.get("Title"),
            }
    except OMDbRateLimitError:
        raise  # Let this propagate up
    except requests.RequestException as e:
        log.error(f"OMDb API error: {e}")

    return None


def create_tv_filename(
    series_title: str,
    season: int,
    episode: int | None,
    episode_title: str | None,
    extension: str,
    lang_suffix: str = "",
) -> str | None:
    """Create a TV show filename: 'Show Name - S01E02 - Episode Title.ext'

    Returns None if episode is None (can't create valid filename).
    """
    if episode is None:
        return None

    clean_title = re.sub(r'[<>:"/\\|?*]', "", series_title)
    ep_code = f"S{season:02d}E{episode:02d}"

    if episode_title:
        # Decode HTML entities from OMDb (e.g., &gt; -> >)
        episode_title = html.unescape(episode_title)
        # Replace / with - (not allowed in Windows filenames)
        episode_title = episode_title.replace("/", "-")
        clean_ep_title = re.sub(r'[<>:"/\\|?*]', "", episode_title)
        clean_ep_title = re.sub(r"\s+", " ", clean_ep_title).strip()
        base = f"{clean_title} - {ep_code} - {clean_ep_title}"
    else:
        base = f"{clean_title} - {ep_code}"

    if lang_suffix:
        return f"{base}{lang_suffix}{extension}"
    return f"{base}{extension}"


def find_media_in_source() -> tuple[list[dict], list[dict]]:
    """Find all media in source directory, separating movies from TV shows."""
    source = Path(config.SOURCE_DIR)
    if not source.exists():
        log.error(f"Source directory does not exist: {source}")
        return [], []

    movies = []
    tv_shows = []

    for item in source.iterdir():
        if item.is_file() and item.suffix.lower() in config.VIDEO_EXTENSIONS:
            if is_still_transferring(item.stem):
                log.info(f"SKIP: Still transferring: {item.name}")
                continue

            # Check if it's a TV show
            tv_info = parse_tv_info(item.stem)
            if tv_info:
                tv_shows.append({
                    "type": "loose",
                    "path": item,
                    "parse_name": item.stem,
                    "tv_info": tv_info,
                })
            else:
                movies.append({
                    "type": "loose",
                    "path": item,
                    "parse_name": item.stem,
                })

        elif item.is_dir():
            if item.name.lower() == "incomplete":
                continue
            if is_still_transferring(item.name):
                log.info(f"SKIP: Still transferring: {item.name}")
                continue

            video_files = get_all_files(item, config.VIDEO_EXTENSIONS)
            if video_files:
                # Check if it's a TV show based on folder name or first video file
                tv_info = parse_tv_info(item.name)
                if not tv_info and video_files:
                    tv_info = parse_tv_info(video_files[0].stem)

                if tv_info:
                    tv_shows.append({
                        "type": "folder",
                        "path": item,
                        "parse_name": item.name,
                        "video_files": video_files,
                        "subtitle_files": get_all_files(item, config.SUBTITLE_EXTENSIONS),
                        "tv_info": tv_info,
                    })
                else:
                    movies.append({
                        "type": "folder",
                        "path": item,
                        "parse_name": item.name,
                        "video_files": video_files,
                        "subtitle_files": get_all_files(item, config.SUBTITLE_EXTENSIONS),
                    })

    return movies, tv_shows


def move_tv_show(show: dict, series_info: dict, dry_run: bool = False) -> bool:
    """Move a TV show episode to the destination with proper naming."""
    series_title = series_info["title"]
    series_year = series_info["year"]
    tv_info = show["tv_info"]
    season = tv_info["season"]
    episode = tv_info["episode"]

    # PTN returns lists for multi-season/episode packs
    if isinstance(season, list):
        season = season[0]
    if isinstance(episode, list):
        episode = episode[0]

    # Try to get episode title from OMDb (only if we have an episode number)
    episode_title = tv_info.get("episode_title")
    if not episode_title and episode:
        ep_info = query_omdb_episode(series_title, season, episode)
        if ep_info:
            episode_title = ep_info.get("episode_title")
            time.sleep(0.15)  # Rate limiting

    # Create folder structure: Show Name (Year)/Season XX/
    series_folder_name = f"{re.sub(r'[<>:\"/\\|?*]', '', series_title)} ({series_year})"
    tv_base = get_tv_dest(series_folder_name)
    season_folder_name = f"Season {season:02d}"
    dest_folder = tv_base / series_folder_name / season_folder_name

    if dry_run:
        if show["type"] == "loose":
            src_file = show["path"]
            dest_name = create_tv_filename(
                series_title, season, episode, episode_title, src_file.suffix.lower()
            )
            if dest_name is None:
                log.info(f"SKIP: No episode number found in: {src_file.name}")
                return False
            dest_file = dest_folder / dest_name
            log.info(f"[DRY RUN] Would move: {src_file.name} -> {dest_file}")
        elif show["type"] == "folder":
            files_found = 0
            for video_file in show["video_files"]:
                # Re-parse each file for correct season and episode number
                file_tv_info = parse_tv_info(video_file.stem) or {}
                file_season = file_tv_info.get("season") or season
                file_episode = file_tv_info.get("episode") or episode
                if isinstance(file_season, list):
                    file_season = file_season[0]
                if isinstance(file_episode, list):
                    file_episode = file_episode[0]

                # Build per-file destination folder based on file's season
                file_season_folder = f"Season {file_season:02d}"
                file_dest_folder = tv_base / series_folder_name / file_season_folder

                # Query OMDb for episode title if we have an episode number
                file_ep_title = file_tv_info.get("episode_title") or episode_title
                if not file_ep_title and file_episode:
                    ep_info = query_omdb_episode(series_title, file_season, file_episode)
                    if ep_info:
                        file_ep_title = ep_info.get("episode_title")
                        time.sleep(0.15)

                dest_name = create_tv_filename(
                    series_title, file_season, file_episode, file_ep_title, video_file.suffix.lower()
                )
                if dest_name is None:
                    log.info(f"  SKIP: No episode number in: {video_file.name}")
                    continue
                files_found += 1
                dest_file = file_dest_folder / dest_name
                log.info(f"[DRY RUN] Would move: {video_file.name} -> {dest_file}")

            for sub_file in show.get("subtitle_files", []):
                stem, suffix = get_file_parts(sub_file)
                lang_suffix = extract_language_suffix(stem)
                file_tv_info = parse_tv_info(stem) or {}
                file_season = file_tv_info.get("season") or season
                file_episode = file_tv_info.get("episode") or episode
                if isinstance(file_season, list):
                    file_season = file_season[0]
                if isinstance(file_episode, list):
                    file_episode = file_episode[0]

                file_season_folder = f"Season {file_season:02d}"
                file_dest_folder = tv_base / series_folder_name / file_season_folder

                dest_name = create_tv_filename(
                    series_title, file_season, file_episode, None, suffix, lang_suffix
                )
                if dest_name is None:
                    continue  # Skip subtitles without episode numbers
                dest_file = file_dest_folder / dest_name
                log.info(f"[DRY RUN] Would move: {sub_file.name} -> {dest_file}")

            if files_found == 0:
                log.info(f"SKIP: No files with episode numbers found in: {show['path'].name}")
                return False

            log.info(f"[DRY RUN] Would delete source folder: {show['path'].name}")
        return True

    # Actual move (not dry run)
    try:
        if show["type"] == "loose":
            src_file = show["path"]
            dest_name = create_tv_filename(
                series_title, season, episode, episode_title, src_file.suffix.lower()
            )
            if dest_name is None:
                log.info(f"SKIP: No episode number found in: {src_file.name}")
                return False
            dest_folder.mkdir(parents=True, exist_ok=True)
            dest_file = dest_folder / dest_name
            if dest_file.exists():
                log.warning(f"SKIP: Destination already exists: {dest_file}")
                return False
            shutil.move(str(src_file), str(dest_file))
            log.info(f"Moved -> {dest_file}")

        elif show["type"] == "folder":
            files_moved = 0
            for video_file in show["video_files"]:
                file_tv_info = parse_tv_info(video_file.stem) or {}
                file_season = file_tv_info.get("season") or season
                file_episode = file_tv_info.get("episode") or episode
                if isinstance(file_season, list):
                    file_season = file_season[0]
                if isinstance(file_episode, list):
                    file_episode = file_episode[0]

                # Build per-file destination folder based on file's season
                file_season_folder = f"Season {file_season:02d}"
                file_dest_folder = tv_base / series_folder_name / file_season_folder

                # Query OMDb for episode title if we have an episode number
                file_ep_title = file_tv_info.get("episode_title") or episode_title
                if not file_ep_title and file_episode:
                    ep_info = query_omdb_episode(series_title, file_season, file_episode)
                    if ep_info:
                        file_ep_title = ep_info.get("episode_title")
                        time.sleep(0.15)

                dest_name = create_tv_filename(
                    series_title, file_season, file_episode, file_ep_title, video_file.suffix.lower()
                )
                if dest_name is None:
                    log.info(f"  SKIP: No episode number in: {video_file.name}")
                    continue
                file_dest_folder.mkdir(parents=True, exist_ok=True)
                dest_file = file_dest_folder / dest_name
                if dest_file.exists():
                    log.warning(f"  SKIP: Destination already exists: {dest_file.name}")
                    continue
                shutil.move(str(video_file), str(dest_file))
                log.info(f"Moved -> {dest_file}")
                files_moved += 1

            for sub_file in show.get("subtitle_files", []):
                stem, suffix = get_file_parts(sub_file)
                lang_suffix = extract_language_suffix(stem)
                file_tv_info = parse_tv_info(stem) or {}
                file_season = file_tv_info.get("season") or season
                file_episode = file_tv_info.get("episode") or episode
                if isinstance(file_season, list):
                    file_season = file_season[0]
                if isinstance(file_episode, list):
                    file_episode = file_episode[0]

                file_season_folder = f"Season {file_season:02d}"
                file_dest_folder = tv_base / series_folder_name / file_season_folder

                dest_name = create_tv_filename(
                    series_title, file_season, file_episode, None, suffix, lang_suffix
                )
                if dest_name is None:
                    continue  # Skip subtitles without episode numbers
                file_dest_folder.mkdir(parents=True, exist_ok=True)
                dest_file = file_dest_folder / dest_name
                if dest_file.exists():
                    continue  # Skip silently for subs
                shutil.move(str(sub_file), str(dest_file))
                log.info(f"Moved -> {dest_file}")

            if files_moved == 0:
                log.info(f"SKIP: No files with episode numbers found in: {show['path'].name}")
                return False

            cleanup_source_folder(show["path"])

        return True

    except Exception as e:
        log.error(f"ERROR moving files: {e}")
        return False


def create_clean_filename(title: str, year: str, extension: str, lang_suffix: str = "") -> str:
    """Create a clean filename from title and year."""
    clean = re.sub(r'[<>:"/\\|?*]', "", title)
    if lang_suffix:
        return f"{clean} ({year}){lang_suffix}{extension}"
    return f"{clean} ({year}){extension}"


def get_file_parts(filepath: Path) -> tuple[str, str]:
    """Get stem and suffix from a file, handling edge cases like '.srt'.

    Returns (stem, suffix) where suffix includes the leading dot.
    For files like '.srt', returns ('', '.srt').
    """
    name = filepath.name
    suffix = filepath.suffix

    # Handle files like ".srt" where pathlib treats it as hidden with no extension
    if not suffix and name.startswith("."):
        return "", name.lower()

    return filepath.stem, suffix.lower()


def extract_language_suffix(filename: str) -> str:
    """Extract language suffix from subtitle filename if present.

    Expects the filename stem (without extension), e.g., "Forced.eng" not "Forced.eng.srt".
    """
    # Known ISO 639-1 and 639-2 codes
    iso_codes = {
        "en", "eng", "es", "spa", "fr", "fre", "fra", "de", "ger", "deu",
        "it", "ita", "pt", "por", "ru", "rus", "zh", "chi", "zho",
        "ja", "jpn", "ko", "kor", "ar", "ara", "hi", "hin", "nl", "dut",
        "nld", "sv", "swe", "no", "nor", "nob", "nb", "nn", "nno",  # Norwegian variants
        "da", "dan", "fi", "fin", "pl", "pol", "tr", "tur", "cs", "cze", "ces",
        "el", "gre", "ell", "he", "heb", "th", "tha", "vi", "vie",
        "id", "ind", "ms", "msa", "ro", "rum", "ron", "hu", "hun",
        "uk", "ukr", "bg", "bul", "hr", "hrv", "sk", "slo", "slk",
        "sl", "slv", "et", "est", "lv", "lav", "lt", "lit",
        "cat", "baq", "fil", "glg", "may", "tam", "tel", "urd",
        "ben", "mar", "guj", "kan", "mal", "pan", "yue",  # Cantonese
    }
    # Full language names and special tags
    full_names = {
        "english", "spanish", "french", "german", "italian", "portuguese",
        "russian", "chinese", "japanese", "korean", "arabic", "hindi",
        "dutch", "swedish", "norwegian", "danish", "finnish", "polish",
        "turkish", "czech", "greek", "hebrew", "thai", "vietnamese",
        "indonesian", "malay", "romanian", "hungarian", "ukrainian",
        "bulgarian", "croatian", "slovak", "slovenian", "brazilian",
        "estonian", "latvian", "lithuanian", "filipino", "catalan",
        "simplified", "traditional", "latin", "european", "canadian",  # Region modifiers
        "forced", "sdh", "hi", "cc",  # Special tags (hi = hearing impaired)
    }

    # Use filename directly - it's already the stem
    stem = filename.strip()
    if not stem:
        return ""

    stem_lower = stem.lower()

    # Check if entire stem is a language code (e.g., "ara" from "ara.srt")
    if stem_lower in iso_codes or stem_lower in full_names:
        return f".{stem_lower}"

    # Split by dots, underscores, and spaces
    parts = re.split(r"[._ ]+", stem_lower)

    # Filter out empty parts
    parts = [p for p in parts if p]
    if not parts:
        return ""

    # Take only the last 1-4 parts that might be language-related
    suffix_parts = []
    for part in reversed(parts[-4:]):
        if part in iso_codes or part in full_names:
            suffix_parts.insert(0, part)
        elif suffix_parts:
            # Stop when we hit a non-language part after finding some
            break

    if suffix_parts:
        return "." + ".".join(suffix_parts)

    return ""


def move_movie(movie: dict, omdb_info: dict, dry_run: bool = False) -> bool:
    """Move a movie to the destination with proper naming."""
    title = omdb_info["title"]
    year = omdb_info["year"]
    dest_folder_name = create_clean_filename(title, year, "")
    movie_base = get_movie_dest(dest_folder_name)
    dest_folder = movie_base / dest_folder_name

    if dest_folder.exists():
        log.info(f"SKIP: Destination already exists: {dest_folder}")
        return False

    if dry_run:
        # Just log what would happen
        if movie["type"] == "loose":
            src_file = movie["path"]
            dest_file = dest_folder / create_clean_filename(title, year, src_file.suffix.lower())
            log.info(f"[DRY RUN] Would move: {src_file.name} -> {dest_file}")
        elif movie["type"] == "folder":
            for video_file in movie["video_files"]:
                dest_file = dest_folder / create_clean_filename(title, year, video_file.suffix.lower())
                log.info(f"[DRY RUN] Would move: {video_file.name} -> {dest_file}")
            for sub_file in movie["subtitle_files"]:
                stem, suffix = get_file_parts(sub_file)
                lang_suffix = extract_language_suffix(stem)
                dest_file = dest_folder / create_clean_filename(title, year, suffix, lang_suffix)
                log.info(f"[DRY RUN] Would move: {sub_file.name} -> {dest_file}")
            log.info(f"[DRY RUN] Would delete source folder: {movie['path'].name}")
        return True

    # Create destination folder
    dest_folder.mkdir(parents=True, exist_ok=True)

    try:
        if movie["type"] == "loose":
            src_file = movie["path"]
            dest_file = dest_folder / create_clean_filename(title, year, src_file.suffix.lower())
            shutil.move(str(src_file), str(dest_file))
            log.info(f"Moved -> {dest_file}")

        elif movie["type"] == "folder":
            for video_file in movie["video_files"]:
                dest_file = dest_folder / create_clean_filename(title, year, video_file.suffix.lower())
                shutil.move(str(video_file), str(dest_file))
                log.info(f"Moved -> {dest_file}")

            for sub_file in movie["subtitle_files"]:
                stem, suffix = get_file_parts(sub_file)
                lang_suffix = extract_language_suffix(stem)
                dest_file = dest_folder / create_clean_filename(title, year, suffix, lang_suffix)
                shutil.move(str(sub_file), str(dest_file))
                log.info(f"Moved -> {dest_file}")

            # Warn if non-video/subtitle files exist before cleanup
            remaining = [
                f for f in movie["path"].rglob("*")
                if f.is_file()
                and f.suffix.lower() not in config.VIDEO_EXTENSIONS + config.SUBTITLE_EXTENSIONS
            ]
            if remaining:
                log.warning(
                    f"Deleting source folder with {len(remaining)} extra file(s): "
                    f"{', '.join(f.name for f in remaining[:5])}"
                )

            cleanup_source_folder(movie["path"])

        return True

    except Exception as e:
        log.error(f"ERROR moving files: {e}")
        if dest_folder.exists() and not any(dest_folder.iterdir()):
            dest_folder.rmdir()
        return False


def cleanup_source_folder(folder: Path) -> None:
    """Remove source folder entirely after moving video/subtitle files."""
    try:
        if folder.is_dir():
            shutil.rmtree(folder)
            log.info(f"Deleted source folder: {folder.name}")
    except Exception as e:
        log.error(f"Could not delete folder {folder}: {e}")


def _progress_file_path() -> Path:
    """Get the path to the library progress file (next to the script)."""
    return Path(__file__).parent / config.LIBRARY_PROGRESS_FILE


def load_library_progress() -> dict:
    """Load library progress from JSON file."""
    path = _progress_file_path()
    if path.exists():
        try:
            data = json.loads(path.read_text())
            return {
                "movies": set(data.get("movies", [])),
                "tv": set(data.get("tv", [])),
            }
        except (json.JSONDecodeError, KeyError):
            log.warning(f"Corrupt progress file, starting fresh: {path}")
    return {"movies": set(), "tv": set()}


def save_library_progress(progress: dict) -> None:
    """Save library progress to JSON file."""
    path = _progress_file_path()
    data = {
        "movies": sorted(progress.get("movies", set())),
        "tv": sorted(progress.get("tv", set())),
    }
    path.write_text(json.dumps(data, indent=2))


def clear_library_progress(mode: str) -> None:
    """Clear library progress for the given mode ('movies', 'tv', or 'all')."""
    if mode == "all":
        path = _progress_file_path()
        if path.exists():
            path.unlink()
            log.info("Cleared all library progress.")
        else:
            log.info("No progress file found.")
        return

    progress = load_library_progress()
    if mode in progress:
        count = len(progress[mode])
        progress[mode] = set()
        save_library_progress(progress)
        log.info(f"Cleared {mode} library progress ({count} entries).")
    else:
        log.info(f"No {mode} progress found.")


def is_properly_named(folder_name: str) -> bool:
    """Check if a folder is already in 'Title (Year)' format."""
    # Match pattern like "Movie Name (2024)" or "Movie Name (2024-2025)"
    return bool(re.match(r"^.+\s+\(\d{4}(-\d{4})?\)$", folder_name))


def rename_library_folder(folder: Path, omdb_info: dict, dry_run: bool = False) -> bool:
    """Rename a library folder and its contents to proper naming."""
    title = omdb_info["title"]
    year = omdb_info["year"]
    new_folder_name = create_clean_filename(title, year, "")
    new_folder = folder.parent / new_folder_name

    # Skip if already named correctly
    if folder.name == new_folder_name:
        log.info(f"  Already correctly named: {folder.name}")
        return False

    # Check if target already exists
    if new_folder.exists() and new_folder != folder:
        log.warning(f"  SKIP: Target folder already exists: {new_folder_name}")
        return False

    if dry_run:
        log.info(f"  [DRY RUN] Would rename folder: {folder.name} -> {new_folder_name}")
        # Show what files would be renamed
        for video_file in get_all_files(folder, config.VIDEO_EXTENSIONS):
            new_name = create_clean_filename(title, year, video_file.suffix.lower())
            if video_file.name != new_name:
                log.info(f"  [DRY RUN] Would rename file: {video_file.name} -> {new_name}")
        return True

    try:
        # First rename files inside the folder
        for video_file in get_all_files(folder, config.VIDEO_EXTENSIONS):
            new_name = create_clean_filename(title, year, video_file.suffix.lower())
            if video_file.name != new_name:
                new_path = video_file.parent / new_name
                video_file.rename(new_path)
                log.info(f"  Renamed file: {video_file.name} -> {new_name}")

        for sub_file in get_all_files(folder, config.SUBTITLE_EXTENSIONS):
            stem, suffix = get_file_parts(sub_file)
            lang_suffix = extract_language_suffix(stem)
            new_name = create_clean_filename(title, year, suffix, lang_suffix)
            if sub_file.name != new_name:
                new_path = sub_file.parent / new_name
                sub_file.rename(new_path)
                log.info(f"  Renamed file: {sub_file.name} -> {new_name}")

        # Then rename the folder itself
        if folder.name != new_folder_name:
            folder.rename(new_folder)
            log.info(f"  Renamed folder: {folder.name} -> {new_folder_name}")

        return True

    except Exception as e:
        log.error(f"  ERROR renaming: {e}")
        return False


def rename_files_in_folder(folder: Path, title: str, year: str, dry_run: bool = False) -> int:
    """Rename files inside a folder to proper naming. Returns count of files renamed."""
    renamed_count = 0

    for video_file in get_all_files(folder, config.VIDEO_EXTENSIONS):
        new_name = create_clean_filename(title, year, video_file.suffix.lower())
        if video_file.name != new_name:
            if dry_run:
                log.info(f"  [DRY RUN] Would rename: {video_file.name} -> {new_name}")
            else:
                new_path = video_file.parent / new_name
                video_file.rename(new_path)
                log.info(f"  Renamed: {video_file.name} -> {new_name}")
            renamed_count += 1

    for sub_file in get_all_files(folder, config.SUBTITLE_EXTENSIONS):
        stem, suffix = get_file_parts(sub_file)
        lang_suffix = extract_language_suffix(stem)
        new_name = create_clean_filename(title, year, suffix, lang_suffix)
        if sub_file.name != new_name:
            if dry_run:
                log.info(f"  [DRY RUN] Would rename: {sub_file.name} -> {new_name}")
            else:
                new_path = sub_file.parent / new_name
                sub_file.rename(new_path)
                log.info(f"  Renamed: {sub_file.name} -> {new_name}")
            renamed_count += 1

    return renamed_count


def process_library(dry_run: bool = False) -> tuple[int, int, int]:
    """Process existing library to rename improperly named folders and files."""
    if dry_run:
        log.info("=== DRY RUN MODE - No files will be renamed ===")

    # Collect folders from all movie drives
    all_folders = []
    for lib_dir in config.MOVIE_DEST_DIRS:
        library = Path(lib_dir)
        if not library.exists():
            log.warning(f"Skipping missing library: {library}")
            continue
        log.info(f"Scanning library: {library}")
        all_folders.extend(sorted(f for f in library.iterdir() if f.is_dir()))

    if not all_folders:
        log.error("No movie library directories found")
        return 0, 0, 0

    progress = load_library_progress()
    prev_done = progress["movies"]

    renamed = 0
    skipped = 0
    already_ok = 0
    prev_skipped = 0

    folders = all_folders
    total = len(folders)

    try:
        for i, folder in enumerate(folders, 1):
            # Skip folders already processed in a previous run
            if folder.name in prev_done:
                prev_skipped += 1
                continue

            log.info(f"[{i}/{total}] Checking: {folder.name}")

            # Parse title and year from folder name
            title, year = parse_filename(folder.name)
            if not title:
                log.info(f"  SKIP: Could not parse title from folder name")
                skipped += 1
                continue

            # If folder is properly named, use its title/year directly for file renaming
            if is_properly_named(folder.name):
                # Extract title and year from the folder name pattern "Title (Year)"
                match = re.match(r"^(.+)\s+\((\d{4})\)$", folder.name)
                if match:
                    title = match.group(1)
                    year = match.group(2)

                    # Check and rename files inside
                    files_renamed = rename_files_in_folder(folder, title, year, dry_run=dry_run)
                    if files_renamed > 0:
                        renamed += 1
                    else:
                        log.info(f"  Files already correctly named")
                        already_ok += 1
                    progress["movies"].add(folder.name)
                    save_library_progress(progress)
                    continue

            log.info(f"  Parsed: title='{title}', year={year or 'none'}")

            # Query OMDb for folders not in correct format
            omdb_info = query_omdb(title, year)
            if not omdb_info:
                log.info(f"  SKIP: No confident OMDb match")
                skipped += 1
                continue

            log.info(f"  OMDb match: {omdb_info['title']} ({omdb_info['year']})")

            # Rename the folder and contents
            if rename_library_folder(folder, omdb_info, dry_run=dry_run):
                renamed += 1
                # Track by new name after rename
                new_name = create_clean_filename(omdb_info["title"], omdb_info["year"], "")
                progress["movies"].add(new_name)
            else:
                already_ok += 1
                progress["movies"].add(folder.name)
            save_library_progress(progress)

    except OMDbRateLimitError as e:
        log.error(f"ABORTING: {e}")
        log.error("Partial progress has been saved. Re-run once the API limit resets.")
    except KeyboardInterrupt:
        log.info("Interrupted. Progress has been saved. Re-run to continue.")

    prefix = "[DRY RUN] " if dry_run else ""
    if prev_skipped > 0:
        log.info(f"{prefix}Skipped {prev_skipped} previously processed folders.")
    log.info(f"{prefix}Library scan complete. Renamed: {renamed}, Skipped: {skipped}, Already OK: {already_ok}")
    return renamed, skipped, already_ok


def rename_tv_files_in_season(
    season_folder: Path, series_title: str, season: int, dry_run: bool = False,
    trust_folder_season: bool = False,
) -> int:
    """Rename episode files inside a season folder. Returns count of files renamed.

    When trust_folder_season=True (matched season folder), always use the folder's season number.
    When trust_folder_season=False (no season subfolders), use each file's own parsed season.
    """
    renamed_count = 0

    # Cache episode titles from OMDb to avoid duplicate calls for subs
    # Key is (season, episode) tuple
    episode_title_cache: dict[tuple[int, int], str | None] = {}

    def get_episode_title(s: int, ep: int) -> str | None:
        """Get episode title, preferring OMDb over PTN (PTN truncates at special chars)."""
        key = (s, ep)
        if key in episode_title_cache:
            return episode_title_cache[key]

        ep_info = query_omdb_episode(series_title, s, ep)
        ep_title = None
        if ep_info:
            ep_title = ep_info.get("episode_title")
            # Skip generic "Episode N" titles
            if ep_title and re.match(r"^Episode \d+$", ep_title):
                ep_title = None
            time.sleep(0.15)

        episode_title_cache[key] = ep_title
        return ep_title

    for video_file in get_all_files(season_folder, config.VIDEO_EXTENSIONS):
        # Parse episode info from the filename
        tv_info = parse_tv_info(video_file.stem)
        if not tv_info or not tv_info.get("episode"):
            continue

        # In a matched season folder, trust the folder's season number (filenames may be wrong)
        # Without season folders, use each file's own parsed season
        file_season = season if trust_folder_season else (tv_info.get("season") or season)
        file_episode = tv_info["episode"]
        if isinstance(file_season, list):
            file_season = file_season[0]
        if isinstance(file_episode, list):
            file_episode = file_episode[0]

        # Extract existing episode title from filename if it's already in our format
        # "Show Name - S01E02 - Episode Title.ext" -> "Episode Title"
        existing_ep_title = None
        ep_title_match = re.match(
            r".+ - S\d{2}E\d{2} - (.+)$", video_file.stem
        )
        if ep_title_match:
            existing_ep_title = ep_title_match.group(1)

        # Always prefer OMDb title (PTN truncates at special chars like ' / = etc.)
        file_ep_title = get_episode_title(file_season, file_episode)
        # Fall back to existing filename title (already clean), then PTN-parsed title
        if not file_ep_title:
            file_ep_title = existing_ep_title or tv_info.get("episode_title")

        new_name = create_tv_filename(
            series_title, file_season, file_episode, file_ep_title, video_file.suffix.lower()
        )
        if new_name and video_file.name != new_name:
            new_path = video_file.parent / new_name
            # Skip if target file already exists (e.g., specials with same episode number)
            if new_path.exists():
                log.warning(f"    SKIP: Target already exists: {new_name}")
                continue
            if dry_run:
                log.info(f"    [DRY RUN] Would rename: {video_file.name} -> {new_name}")
            else:
                video_file.rename(new_path)
                log.info(f"    Renamed: {video_file.name} -> {new_name}")
            renamed_count += 1

    for sub_file in get_all_files(season_folder, config.SUBTITLE_EXTENSIONS):
        stem, suffix = get_file_parts(sub_file)
        tv_info = parse_tv_info(stem)
        if not tv_info or not tv_info.get("episode"):
            continue

        file_season = season if trust_folder_season else (tv_info.get("season") or season)
        file_episode = tv_info["episode"]
        if isinstance(file_season, list):
            file_season = file_season[0]
        if isinstance(file_episode, list):
            file_episode = file_episode[0]
        lang_suffix = extract_language_suffix(stem)

        new_name = create_tv_filename(
            series_title, file_season, file_episode, None, suffix, lang_suffix
        )
        if new_name and sub_file.name != new_name:
            new_path = sub_file.parent / new_name
            if new_path.exists():
                continue  # Skip silently for subs
            if dry_run:
                log.info(f"    [DRY RUN] Would rename: {sub_file.name} -> {new_name}")
            else:
                sub_file.rename(new_path)
                log.info(f"    Renamed: {sub_file.name} -> {new_name}")
            renamed_count += 1

    return renamed_count


def process_tv_library(dry_run: bool = False) -> tuple[int, int, int]:
    """Process existing TV library to rename folders and episode files."""
    if dry_run:
        log.info("=== DRY RUN MODE - No files will be renamed ===")

    # Collect folders from all TV drives
    all_folders = []
    for lib_dir in config.TV_DEST_DIRS:
        library = Path(lib_dir)
        if not library.exists():
            log.warning(f"Skipping missing TV library: {library}")
            continue
        log.info(f"Scanning TV library: {library}")
        all_folders.extend(sorted(f for f in library.iterdir() if f.is_dir()))

    if not all_folders:
        log.error("No TV library directories found")
        return 0, 0, 0

    progress = load_library_progress()
    prev_done = progress["tv"]

    renamed = 0
    skipped = 0
    already_ok = 0
    prev_skipped = 0

    folders = all_folders
    total = len(folders)

    try:
        for i, folder in enumerate(folders, 1):
            # Skip folders already processed in a previous run
            if folder.name in prev_done:
                prev_skipped += 1
                continue

            log.info(f"[{i}/{total}] Checking: {folder.name}")

            # Check if folder already has year in "Show Name (Year)" format
            match = re.match(r"^(.+)\s+\((\d{4})\)$", folder.name)
            if match:
                series_title = match.group(1)
                series_year = match.group(2)
            else:
                # Need to look up the show on OMDb to get the year
                parsed_title = folder.name
                # Clean common patterns from folder name
                parsed_title = re.sub(r"\s*\(\d{4}p?\)", "", parsed_title)  # Remove resolution
                parsed_title = clean_title(parsed_title)

                series_info = query_omdb_series(parsed_title)
                if not series_info:
                    log.info(f"  SKIP: No OMDb match for series: {parsed_title}")
                    skipped += 1
                    continue

                series_title = series_info["title"]
                series_year = series_info["year"]
                log.info(f"  OMDb match: {series_title} ({series_year})")

                # Rename the folder to include the year
                new_folder_name = f"{re.sub(r'[<>:\"/\\|?*]', '', series_title)} ({series_year})"
                new_folder = folder.parent / new_folder_name

                if new_folder.exists() and new_folder != folder:
                    log.warning(f"  SKIP: Target folder already exists: {new_folder_name}")
                    skipped += 1
                    continue

                if dry_run:
                    log.info(f"  [DRY RUN] Would rename folder: {folder.name} -> {new_folder_name}")
                else:
                    folder.rename(new_folder)
                    log.info(f"  Renamed folder: {folder.name} -> {new_folder_name}")
                    folder = new_folder  # Update reference for file processing below

            # Process season folders inside - match various naming patterns:
            # "Season 01", "Season 1", "Show Name Season 1", "Show Name (2010) Season 1"
            season_folders = []
            for f in sorted(folder.iterdir()):
                if not f.is_dir():
                    continue
                season_match = re.search(r"Season\s+(\d+)", f.name, re.IGNORECASE)
                if season_match:
                    season_folders.append((f, int(season_match.group(1))))

            if not season_folders:
                # Maybe files are directly in the show folder (no season subfolders)
                video_files = get_all_files(folder, config.VIDEO_EXTENSIONS)
                if video_files:
                    # No season folder - use each file's own parsed season
                    files_renamed = rename_tv_files_in_season(
                        folder, series_title, 1, dry_run, trust_folder_season=False
                    )
                    if files_renamed > 0:
                        renamed += 1
                    else:
                        already_ok += 1
                else:
                    already_ok += 1
                progress["tv"].add(folder.name)
                save_library_progress(progress)
                continue

            show_had_renames = False
            for season_folder, season_num in season_folders:
                # Rename non-standard season folders (e.g. "Show Name Season 1" -> "Season 01")
                standard_name = f"Season {season_num:02d}"
                if season_folder.name != standard_name:
                    new_season_path = season_folder.parent / standard_name
                    if new_season_path.exists() and new_season_path != season_folder:
                        log.warning(f"  SKIP season folder rename: {standard_name} already exists")
                    elif dry_run:
                        log.info(f"  [DRY RUN] Would rename season folder: {season_folder.name} -> {standard_name}")
                    else:
                        season_folder.rename(new_season_path)
                        log.info(f"  Renamed season folder: {season_folder.name} -> {standard_name}")
                        season_folder = new_season_path

                log.info(f"  Season {season_num:02d}:")
                files_renamed = rename_tv_files_in_season(
                    season_folder, series_title, season_num, dry_run, trust_folder_season=True
                )
                if files_renamed > 0:
                    show_had_renames = True

            if show_had_renames:
                renamed += 1
            else:
                log.info(f"  Files already correctly named")
                already_ok += 1

            progress["tv"].add(folder.name)
            save_library_progress(progress)

    except OMDbRateLimitError as e:
        log.error(f"ABORTING: {e}")
        log.error("Partial progress has been saved. Re-run once the API limit resets.")
    except KeyboardInterrupt:
        log.info("Interrupted. Progress has been saved. Re-run to continue.")

    prefix = "[DRY RUN] " if dry_run else ""
    if prev_skipped > 0:
        log.info(f"{prefix}Skipped {prev_skipped} previously processed folders.")
    log.info(f"{prefix}TV library scan complete. Renamed: {renamed}, Skipped: {skipped}, Already OK: {already_ok}")
    return renamed, skipped, already_ok


def process_movies_list(movies: list[dict], quiet: bool = False, dry_run: bool = False) -> tuple[int, int]:
    """Process a list of movies. Returns (processed, skipped) counts."""
    processed = 0
    skipped = 0

    for movie in movies:
        parse_name = movie["parse_name"]
        log.info(f"[MOVIE] Found: {parse_name}")

        # Parse filename
        title, year = parse_filename(parse_name)
        if not title:
            log.info(f"SKIP: Could not parse title from: {parse_name}")
            skipped += 1
            continue

        log.info(f"  Parsed: title='{title}', year={year or 'none'}")

        # Query OMDb (match validation happens inside query_omdb)
        omdb_info = query_omdb(title, year)
        if not omdb_info:
            log.info(f"SKIP: No confident OMDb match for: {title} ({year or 'no year'})")
            skipped += 1
            continue

        log.info(f"  OMDb match: {omdb_info['title']} ({omdb_info['year']})")

        # Move the movie
        if move_movie(movie, omdb_info, dry_run=dry_run):
            processed += 1
        else:
            skipped += 1

    return processed, skipped


def process_tv_shows_list(tv_shows: list[dict], quiet: bool = False, dry_run: bool = False) -> tuple[int, int]:
    """Process a list of TV shows. Returns (processed, skipped) counts."""
    processed = 0
    skipped = 0

    for show in tv_shows:
        parse_name = show["parse_name"]
        tv_info = show["tv_info"]
        log.info(f"[TV] Found: {parse_name}")

        title = tv_info.get("title")
        if not title:
            log.info(f"SKIP: Could not parse title from: {parse_name}")
            skipped += 1
            continue

        season = tv_info.get("season", 1)
        episode = tv_info.get("episode")

        # PTN returns lists for multi-season/episode packs (e.g. "Season 1-12")
        if isinstance(season, list):
            season = season[0]
        if isinstance(episode, list):
            episode = episode[0]

        ep_str = f"{episode:02d}" if episode else "??"
        log.info(f"  Parsed: title='{title}', S{season:02d}E{ep_str}")

        # Query OMDb for series
        series_info = query_omdb_series(title)
        if not series_info:
            log.info(f"SKIP: No OMDb match for series: {title}")
            skipped += 1
            continue

        log.info(f"  OMDb match: {series_info['title']} ({series_info['year']})")

        # Move the TV show
        if move_tv_show(show, series_info, dry_run=dry_run):
            processed += 1
        else:
            skipped += 1

    return processed, skipped


def process_all(quiet: bool = False, dry_run: bool = False) -> tuple[int, int, int, int]:
    """Process all media in source directory. Returns (movies_processed, movies_skipped, tv_processed, tv_skipped)."""
    if not quiet:
        log.info(f"Scanning {config.SOURCE_DIR}...")
        if dry_run:
            log.info("=== DRY RUN MODE - No files will be moved ===")

    movies, tv_shows = find_media_in_source()

    movies_processed = 0
    movies_skipped = 0
    tv_processed = 0
    tv_skipped = 0

    try:
        if movies:
            movies_processed, movies_skipped = process_movies_list(movies, quiet, dry_run)

        if tv_shows:
            tv_processed, tv_skipped = process_tv_shows_list(tv_shows, quiet, dry_run)

        if not movies and not tv_shows:
            if not quiet:
                log.info("No media found.")
    except OMDbRateLimitError as e:
        log.error(f"ABORTING: {e}")
        log.error("Partial progress has been saved. Re-run once the API limit resets.")

    total_processed = movies_processed + tv_processed
    total_skipped = movies_skipped + tv_skipped

    if total_processed > 0 or total_skipped > 0:
        prefix = "[DRY RUN] " if dry_run else ""
        log.info(f"{prefix}Scan complete. Movies: {movies_processed}/{movies_processed + movies_skipped}, "
                 f"TV: {tv_processed}/{tv_processed + tv_skipped}")

    return movies_processed, movies_skipped, tv_processed, tv_skipped


def run_once(dry_run: bool = False) -> None:
    """Run the renamer once."""
    process_all(quiet=False, dry_run=dry_run)


def run_watch() -> None:
    """Run the renamer in watch mode."""
    log.info(f"Watching {config.SOURCE_DIR} (polling every {config.POLL_INTERVAL}s)")
    log.info("Press Ctrl+C to stop")

    heartbeat_interval = 300  # 5 minutes
    last_heartbeat = time.time()

    try:
        while True:
            process_all(quiet=True)

            if time.time() - last_heartbeat >= heartbeat_interval:
                log.info("Still watching...")
                last_heartbeat = time.time()

            time.sleep(config.POLL_INTERVAL)
    except KeyboardInterrupt:
        log.info("Watch mode stopped.")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Rename and organize movies and TV shows from download folder to library."
    )
    parser.add_argument(
        "--watch", "-w",
        action="store_true",
        help=f"Run in watch mode, scanning every {config.POLL_INTERVAL} seconds",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Preview what would happen without moving any files",
    )
    parser.add_argument(
        "--library", "-l",
        action="store_true",
        help="Scan and rename existing movie library folders to proper naming format",
    )
    parser.add_argument(
        "--library-tv",
        action="store_true",
        help="Scan and rename existing TV show library folders and episodes",
    )
    parser.add_argument(
        "--library-reset",
        choices=["movies", "tv", "all"],
        help="Clear library progress tracking so folders are re-processed from scratch",
    )
    args = parser.parse_args()

    # Library reset mode
    if args.library_reset:
        clear_library_progress(args.library_reset)
        return

    if args.watch and args.dry_run:
        log.error("Cannot combine --watch and --dry-run")
        return

    if args.watch and (args.library or args.library_tv):
        log.error("Cannot combine --watch and --library/--library-tv")
        return

    # Library mode - process existing movie library (scans all drives internally)
    if args.library:
        process_library(dry_run=args.dry_run)
        return

    # TV Library mode - process existing TV library (scans all drives internally)
    if args.library_tv:
        process_tv_library(dry_run=args.dry_run)
        return

    # Normal mode - verify directories exist
    if not Path(config.SOURCE_DIR).exists():
        log.error(f"Source directory does not exist: {config.SOURCE_DIR}")
        return

    # Create primary destination dirs if needed
    if not Path(config.MOVIE_DEST_DIR).exists():
        log.info(f"Creating movie destination directory: {config.MOVIE_DEST_DIR}")
        Path(config.MOVIE_DEST_DIR).mkdir(parents=True, exist_ok=True)

    if not Path(config.TV_DEST_DIR).exists():
        log.info(f"Creating TV destination directory: {config.TV_DEST_DIR}")
        Path(config.TV_DEST_DIR).mkdir(parents=True, exist_ok=True)

    # Log which additional drives are available
    for drive_dir in config.MOVIE_DEST_DIRS:
        if drive_dir != config.MOVIE_DEST_DIR and Path(drive_dir).exists():
            log.info(f"Additional movie drive available: {drive_dir}")
    for drive_dir in config.TV_DEST_DIRS:
        if drive_dir != config.TV_DEST_DIR and Path(drive_dir).exists():
            log.info(f"Additional TV drive available: {drive_dir}")

    if args.watch:
        run_watch()
    else:
        run_once(dry_run=args.dry_run)


if __name__ == "__main__":
    main()