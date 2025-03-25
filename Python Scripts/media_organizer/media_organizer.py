import os
import re
import json
import shutil
import requests
import argparse
import time
import difflib
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any, Set
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Optional imports for LLM integration
try:
    import ollama

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama not available. LLM features will be disabled.")

# Paths
MAIN_DIR = r"I:\share-downloads\complete"  # Source directory to scan
MOVIES_DIR = r"I:\Media3\Movies"  # Destination for movie files
TV_DIR = r"I:\Media3\TV"  # Destination for TV show files

# API configuration
API_URL = "https://imdb.iamidiotareyoutoo.com/search"

# Global configuration
DRY_RUN = False  # Set to False to actually move files, True for simulation only
USE_LLM = True  # Set to True to use LLM for title cleaning and validation
LLM_MODEL = "gemma3:1b"  # Model to use with Ollama


@dataclass
class MediaInfo:
    """Class to store media information"""
    title: str
    year: Optional[str] = None
    content_type: Optional[str] = None  # "movie" or "tv"
    season: Optional[int] = None
    season_range: Optional[Tuple[int, int]] = None
    confidence: float = 0.0

    @property
    def formatted_title(self) -> str:
        """Return title with year in parentheses if available"""
        if self.year:
            return f"{self.title} ({self.year})"
        return self.title


class MediaPattern:
    """Static patterns used for media detection"""
    # Common patterns
    YEAR = r'\b(19\d{2}|20\d{2})\b'
    YEAR_PARENTHESES = r'\((\d{4})\)'
    SEASON = r'(?:S|Season\s*)(\d{1,2})'
    SEASON_RANGE_1 = r'(?:S|Season\s*)(\d{1,2})[-_](?:S|Season\s*)(\d{1,2})'  # Season 1-3 or S01-S03
    SEASON_RANGE_2 = r'(?:S|Season\s*)(\d{1,2})[-_ ](\d{1,2})'  # Season 1-3 or S01-03
    EPISODE = r'(?:E|Episode\s*)(\d{1,2})'

    # Expanded technical terms
    TECHNICAL = r'(?:1080p|720p|2160p|4k|HDR|HEVC|BluRay|WEB-DL|WEBRip|x264|x265|10bit|REMUX|AMZN|DSNP|HULU|' \
                r'HDRip|UHD|BDRip|DVDRip|XviD|DTS|DD5\.1|DD[0-9](?:\.[0-9])?|Dolby|DDP5\.1|EAC3|AAC|' \
                r'IMAX|HDR10|HDR10\+|Atmos|TrueHD|Blu-Ray|Bluray|H264|H265|H\.264|H\.265|MPEG|Mixed|READ\.NFO|NFO|' \
                r'Cropped|NoAds|INT|Complete|Silence|DL|WEB)'

    RELEASE_GROUP = r'(?:\[.*?\]|\-[^\-]*$|\([^(]*\b(?:RARbg|YIFY|YTS|EZTV)\b[^)]*\))'

    # TV show indicators
    TV_INDICATORS = [
        r'(?:S|Season\s*)\d{1,2}',
        r'S\d{1,2}[-_]S\d{1,2}',  # Season range like S01-S03
        r'Season\s*\d+-\d+',  # Season range like Season 1-3
        r'(?:S|Season\s*)\d{1,2}[-_ ]\d{1,2}',  # S01-03 or Season 1-3
        r'Episodes?',
        r'Complete\s+Series',
        r'\bSeason\b',
        r'\bSeries\b',
        r'TVShow',
        r'TV Show',
        r'TV Series'
    ]

    # Words to be removed from titles for better comparison
    TITLE_NOISE_WORDS = [
        'complete', 'web', 'webrip', 'web-dl', 'series', 'season', 'episode',
        'bluray', 'dvdrip', 'bdrip', 'hdtv', 'rip', 'xvid', 'divx', 'x264', 'x265',
        'h264', 'h265', 'hevc', '1080p', '720p', '2160p', '4k', 'uhd', 'hdr',
        'multi', 'ac3', 'aac', 'mp3', 'dd5.1', 'dd5', 'dts', 'hd', 'hq',
        'proper', 'repack', 'extended', 'unrated', 'dubbed', 'subbed', 'sub',
        'internal', 'retail', 'read.nfo', 'nfo', 'yify', 'yts', 'eztv', 'rarbg',
        'release', 'ddp5.1', 'ddp5', 'atmos', 'truehd', 'sdr', 'hdr10', 'dolby',
        'vision', 'dv', 'imax', 'enhanced', 'edition', 'cut', 'directors', "6CH"
        'remastered', 'criterion'
    ]


class MediaOperationTracker:
    """Track media operations for summary reporting"""

    def __init__(self):
        self.tv_operations = []
        self.movie_operations = []
        self.ignored_folders = []
        self.errors = []

    def add_tv_operation(self, source: str, destination: str, season: Optional[int] = None):
        """Add a TV show move operation"""
        self.tv_operations.append({
            "source": source,
            "destination": destination,
            "season": season
        })

    def add_movie_operation(self, source: str, destination: str):
        """Add a movie move operation"""
        self.movie_operations.append({
            "source": source,
            "destination": destination
        })

    def add_ignored_folder(self, folder: str):
        """Add an ignored folder"""
        self.ignored_folders.append(folder)

    def add_error(self, source: str, error_msg: str):
        """Add an error"""
        self.errors.append({
            "source": source,
            "error": error_msg
        })

    def get_summary(self) -> str:
        """Generate a summary of all operations"""
        lines = []
        lines.append("\n=== MEDIA ORGANIZER SUMMARY ===\n")

        # Ignored folders
        if self.ignored_folders:
            lines.append(f"Ignored {len(self.ignored_folders)} system folders:")
            for folder in self.ignored_folders:
                lines.append(f"  - {folder}")
            lines.append("")

        # Movies
        if self.movie_operations:
            lines.append(f"Movies ({len(self.movie_operations)}):")
            for op in self.movie_operations:
                lines.append(f"  - {os.path.basename(op['source'])} → {os.path.basename(op['destination'])}")
            lines.append("")

        # TV Shows
        if self.tv_operations:
            # Group by destination
            shows = {}
            for op in self.tv_operations:
                dest_show = os.path.basename(op['destination'])
                if dest_show not in shows:
                    shows[dest_show] = []
                shows[dest_show].append(op)

            lines.append(f"TV Shows ({len(shows)}) with {len(self.tv_operations)} total seasons/operations:")
            for show, operations in shows.items():
                lines.append(f"  - {show}:")
                for op in operations:
                    if op['season']:
                        lines.append(f"    Season {op['season']}: {os.path.basename(op['source'])}")
                    else:
                        lines.append(f"    Complete Series: {os.path.basename(op['source'])}")
            lines.append("")

        # Errors
        if self.errors:
            lines.append(f"Errors ({len(self.errors)}):")
            for error in self.errors:
                lines.append(f"  - {os.path.basename(error['source'])}: {error['error']}")
            lines.append("")

        return "\n".join(lines)


class MediaCleaner:
    """Main class for cleaning media titles and detecting types"""

    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm and OLLAMA_AVAILABLE

    def clean_title(self, folder_name: str) -> MediaInfo:
        """Clean the title and extract media information from folder name"""
        # First try regex-based extraction
        info = self._regex_extract(folder_name)

        # If LLM is enabled and regex confidence is low, use LLM
        if self.use_llm and info.confidence < 0.7:
            llm_info = self._llm_extract(folder_name)

            # Choose the better result based on confidence
            if llm_info and llm_info.confidence > info.confidence:
                logger.info(f"Using LLM result: {llm_info.formatted_title} (confidence: {llm_info.confidence:.2f})")
                return llm_info

        return info

    def _regex_extract(self, folder_name: str) -> MediaInfo:
        """Extract media information using regex patterns"""
        # Start with a basic detection of whether it's likely a TV show
        is_tv = any(re.search(pattern, folder_name, re.IGNORECASE) for pattern in MediaPattern.TV_INDICATORS)

        # First check for season ranges
        season_range = None
        start_season = None
        end_season = None

        # Check for Season 1-3 or S01-S03 pattern
        range_match = re.search(MediaPattern.SEASON_RANGE_1, folder_name, re.IGNORECASE)
        if range_match:
            start_season = int(range_match.group(1))
            end_season = int(range_match.group(2))
            season_range = (start_season, end_season)
            logger.debug(f"Found season range pattern 1: {start_season}-{end_season}")
        else:
            # Check for Season 1-3 or S01-03 pattern
            range_match = re.search(MediaPattern.SEASON_RANGE_2, folder_name, re.IGNORECASE)
            if range_match:
                start_season = int(range_match.group(1))
                end_season = int(range_match.group(2))
                season_range = (start_season, end_season)
                logger.debug(f"Found season range pattern 2: {start_season}-{end_season}")

        # First check if the title is already in a good format with parenthesized year
        # Pattern: "Title Name (2023) ..."
        clean_title_match = re.match(r'^(.*?\(\d{4}\)).*', folder_name, re.IGNORECASE)

        if clean_title_match:
            # We have a clean title with year already
            base_title = clean_title_match.group(1).strip()

            # Check if it's a reasonably clean title (not just a year)
            if len(base_title) > 7:  # e.g. "X (2023)" is 8 chars
                # Extract the year from the parentheses
                year_match = re.search(r'\((\d{4})\)', base_title)
                year = year_match.group(1) if year_match else None

                # Extract the title without the year for cleaner storage
                title_without_year = re.sub(r'\s*\(\d{4}\)\s*$', '', base_title).strip()

                # Extract season
                season = None
                if is_tv and not season_range:
                    # Look for season after the clean title part
                    after_title = folder_name[len(base_title):].strip()
                    season_match = re.search(MediaPattern.SEASON, after_title, re.IGNORECASE)
                    if season_match:
                        try:
                            season = int(season_match.group(1))
                        except (ValueError, IndexError):
                            pass
                elif season_range:
                    # Use the first season from the range
                    season = start_season

                # If we have a good quality extraction, return with high confidence
                return MediaInfo(
                    title=self._deep_clean_title(title_without_year),  # Store title without year
                    year=year,
                    content_type="tv" if is_tv else "movie",
                    season=season,
                    season_range=season_range,
                    confidence=0.9  # High confidence because the title was already well-formed
                )

        # If we're here, we need to do a more complex extraction
        # Extract the title by removing technical specs
        title = folder_name

        # First remove everything in parentheses that looks like technical specs
        title = re.sub(r'\((?:\d+p|WEB-DL|BluRay|HEVC|x265|x264|EAC3|AAC|DD5\.1|10bit)[^\)]*\)', '', title,
                       flags=re.IGNORECASE)

        # Remove other technical specs
        title = re.sub(MediaPattern.TECHNICAL, '', title, flags=re.IGNORECASE)

        # Remove release group information
        title = re.sub(MediaPattern.RELEASE_GROUP, '', title, flags=re.IGNORECASE)

        # Replace dots, underscores, dashes with spaces
        title = re.sub(r'[._-]', ' ', title)

        # Remove extra spaces and trim
        title = re.sub(r'\s+', ' ', title).strip()

        # For TV shows, try to clean up "Season X" or "SXX" from the title
        if is_tv:
            # Remove season identifiers from the title
            title = re.sub(r'Season\s*\d+(?:-\d+)?', '', title, flags=re.IGNORECASE).strip()
            title = re.sub(r'S\d{1,2}(?:-S\d{1,2})?', '', title, flags=re.IGNORECASE).strip()
            title = re.sub(r'S\d{1,2}[-_ ]\d{1,2}', '', title, flags=re.IGNORECASE).strip()

        # Extract year
        year = None
        year_match = re.search(MediaPattern.YEAR_PARENTHESES, title)
        if year_match:
            year = year_match.group(1)
            # Remove the year from the title to avoid duplication
            title = re.sub(r'\s*\(\d{4}\)\s*', ' ', title).strip()
        else:
            year_match = re.search(MediaPattern.YEAR, title)
            if year_match:
                year = year_match.group(1)
                # Remove the year from the title if it's not already in parentheses
                title = re.sub(MediaPattern.YEAR, '', title).strip()

        # Extract season number if it's a TV show
        season = None
        if is_tv and not season_range:
            season_match = re.search(MediaPattern.SEASON, folder_name, re.IGNORECASE)
            if season_match:
                try:
                    season = int(season_match.group(1))
                except (ValueError, IndexError):
                    pass
        elif season_range:
            # Use the first season from the range
            season = start_season

        # Clean up any remaining parentheses if no year
        if not year:
            title = re.sub(r'\(\s*\)', '', title).strip()

        # Deep clean the title for better quality
        title = self._deep_clean_title(title)

        # Calculate confidence based on quality of extraction
        confidence = 0.5  # Start with moderate confidence

        # Higher confidence if we have a year
        if year:
            confidence += 0.2

        # Higher confidence for TV show with season
        if is_tv and (season or season_range):
            confidence += 0.1

        # Higher confidence if the title looks clean (no remaining technical specs)
        if not re.search(MediaPattern.TECHNICAL, title, re.IGNORECASE):
            confidence += 0.1

        return MediaInfo(
            title=title,
            year=year,
            content_type="tv" if is_tv else "movie",
            season=season,
            season_range=season_range,
            confidence=confidence
        )

    def _deep_clean_title(self, title: str) -> str:
        """Perform deeper cleaning on a title to remove all technical terms and noise words"""
        # Remove leading/trailing spaces
        title = title.strip()

        # Remove any remaining technical terms
        title = re.sub(MediaPattern.TECHNICAL, '', title, flags=re.IGNORECASE)

        # Remove any of the noise words that might be in the title
        for noise_word in MediaPattern.TITLE_NOISE_WORDS:
            title = re.sub(r'\b' + re.escape(noise_word) + r'\b', '', title, flags=re.IGNORECASE)

        # Remove any doubled-up spaces
        title = re.sub(r'\s+', ' ', title).strip()

        # Remove any empty parentheses
        title = re.sub(r'\(\s*\)', '', title).strip()

        # Remove any empty brackets
        title = re.sub(r'\[\s*\]', '', title).strip()

        # Remove any duplicate spaces
        title = re.sub(r'\s+', ' ', title).strip()

        return title

    def _llm_extract(self, folder_name: str) -> Optional[MediaInfo]:
        """Use LLM to extract media information"""
        if not self.use_llm or not OLLAMA_AVAILABLE:
            return None

        prompt = f"""
Analyze this media filename and extract the following information:
1. Cleaned Title (without technical specs or year in the title)
2. Release Year
3. Content Type (TV or MOVIE)
4. Season Number (if applicable)
5. Season Range (if applicable, e.g. Season 1-3 should return [1, 3])
6. Confidence (how confident are you in this analysis, 0-100%)

Format your response as JSON exactly like this:
{{
  "title": "Cleaned Title",
  "year": "YYYY",
  "type": "TV or MOVIE",
  "season": "Season number or null",
  "season_range": [start_season, end_season] or null,
  "confidence": "Number between 0-100"
}}

Original filename: {folder_name}

Examples:
"The.Matrix.1999.1080p.BluRay.x264" -> {{"title": "The Matrix", "year": "1999", "type": "MOVIE", "season": null, "season_range": null, "confidence": 95}}
"Reacher.S01.1080p.x265-ELiTE" -> {{"title": "Reacher", "year": "2022", "type": "TV", "season": "1", "season_range": null, "confidence": 90}}
"Avatar.2009.BluRay" -> {{"title": "Avatar", "year": "2009", "type": "MOVIE", "season": null, "season_range": null, "confidence": 95}}
"Obi-Wan.Kenobi.S01.2022.DSNP" -> {{"title": "Obi-Wan Kenobi", "year": "2022", "type": "TV", "season": "1", "season_range": null, "confidence": 95}}
"The Bear (2022) Season 2" -> {{"title": "The Bear", "year": "2022", "type": "TV", "season": "2", "season_range": null, "confidence": 98}}
"The Orville (2017) Season 1-3" -> {{"title": "The Orville", "year": "2017", "type": "TV", "season": "1", "season_range": [1, 3], "confidence": 95}}
"Twisted Metal - Season 1 (2023)" -> {{"title": "Twisted Metal", "year": "2023", "type": "TV", "season": "1", "season_range": null, "confidence": 95}}
"""
        try:
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.1, 'num_predict': 300}
            )

            content = response['message']['content'].strip()
            logger.debug(f"LLM response: {content}")

            # Try to find JSON in the response
            json_match = re.search(r'({.*})', content, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(1))

                    # Convert values to appropriate types
                    title = result.get("title", "")
                    year = result.get("year", "")
                    content_type = result.get("type", "").lower()
                    season_str = result.get("season")
                    season_range = result.get("season_range")
                    confidence_str = result.get("confidence", "50")

                    # Normalize content type
                    if "tv" in content_type:
                        content_type = "tv"
                    elif "movie" in content_type:
                        content_type = "movie"

                    # Convert season to int if possible
                    season = None
                    if season_str and season_str != "null":
                        try:
                            season = int(season_str)
                        except ValueError:
                            pass

                    # Convert season_range to tuple if possible
                    season_range_tuple = None
                    if season_range and isinstance(season_range, list) and len(season_range) == 2:
                        try:
                            start_season = int(season_range[0])
                            end_season = int(season_range[1])
                            season_range_tuple = (start_season, end_season)
                        except (ValueError, TypeError):
                            pass

                    # Convert confidence to float
                    try:
                        confidence = float(confidence_str) / 100
                    except ValueError:
                        confidence = 0.5

                    return MediaInfo(
                        title=title,
                        year=year,
                        content_type=content_type,
                        season=season,
                        season_range=season_range_tuple,
                        confidence=confidence
                    )
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON from LLM response: {content}")

            return None
        except Exception as e:
            logger.error(f"Error using LLM: {str(e)}")
            return None


class ImdbAPIClient:
    """Client for searching the IMDB API"""

    def __init__(self, api_url: str, use_llm: bool = False):
        self.api_url = api_url
        self.use_llm = use_llm and OLLAMA_AVAILABLE

    def search(self, media_info: MediaInfo, original_title: str) -> Optional[MediaInfo]:
        """Search the IMDB API for media information"""

        # Generate search variations
        variations = self._generate_search_variations(media_info, original_title)

        # Search for each variation
        best_match = None
        best_confidence = 0.0

        for query in variations:
            result = self._search_api(query, media_info.content_type)

            if result:
                # Calculate match confidence
                confidence = self._calculate_confidence(result, media_info, original_title)

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = result

                    # If we have a high confidence match, no need to try more variations
                    if confidence > 0.8:
                        break

        # If we found a match with good confidence, return it
        if best_match and best_confidence > 0.5:
            api_result = best_match["item"]

            # Extract information from the API result
            title = api_result.get("#TITLE", "")
            year = api_result.get("#YEAR", "")
            content_type = best_match.get("type", media_info.content_type)

            logger.info(f"API match: {title} ({year}) - {content_type} with confidence {best_confidence:.2f}")

            return MediaInfo(
                title=title,
                year=year,
                content_type=content_type,
                season=media_info.season,  # Keep the detected season
                season_range=media_info.season_range,  # Keep the detected season range
                confidence=best_confidence
            )

        # If no good match found, return the original info
        return media_info

    def _generate_search_variations(self, media_info: MediaInfo, original_title: str) -> List[str]:
        """Generate search variations for the title"""
        variations = [media_info.title]

        # Basic variations
        variations.append(re.sub(r'^(The|A|An) ', '', media_info.title, flags=re.IGNORECASE))

        # Add version with year if we have it
        if media_info.year:
            variations.append(f"{media_info.title} {media_info.year}")

        # Add version with season if it's a TV show
        if media_info.content_type == "tv" and media_info.season:
            variations.append(f"{media_info.title} Season {media_info.season}")
            variations.append(f"{media_info.title} S{media_info.season:02d}")

        # Add version with season range if it's a TV show with season range
        if media_info.content_type == "tv" and media_info.season_range:
            start, end = media_info.season_range
            variations.append(f"{media_info.title} Season {start}-{end}")
            variations.append(f"{media_info.title} S{start:02d}-S{end:02d}")

        # Generate more variations with LLM if enabled
        if self.use_llm:
            llm_variations = self._llm_generate_variations(media_info.title, original_title)
            if llm_variations:
                variations.extend(llm_variations)

        # Remove duplicates while preserving order
        variations = list(dict.fromkeys(variations))

        return variations

    def _llm_generate_variations(self, title: str, original_title: str) -> List[str]:
        """Use LLM to generate search variations"""
        if not self.use_llm or not OLLAMA_AVAILABLE:
            return []

        prompt = f"""
Generate 3 alternative search terms for this media title that would help find it on IMDB.
Return ONLY a JSON array of strings with no explanation.

Original filename: {original_title}
Cleaned title: {title}

For example:
Original: "The.Matrix.1999.1080p.BluRay"
Cleaned: "The Matrix"
Response: ["The Matrix", "Matrix 1999", "The Matrix Sci-Fi"]

Original: "Reacher.S01.1080p.x265"
Cleaned: "Reacher" 
Response: ["Reacher", "Jack Reacher TV", "Alan Ritchson Reacher"]

Original: "The Orville (2017) Season 1-3 S01-S03 (1080p Mixed WEB-DL x265 HEVC 10bit EAC3 5.1 Silence)"
Cleaned: "The Orville"
Response: ["The Orville", "The Orville TV Series", "Seth MacFarlane The Orville"]
"""
        try:
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.3, 'num_predict': 100}
            )

            content = response['message']['content'].strip()
            logger.debug(f"LLM variations response: {content}")

            # Try to extract JSON array
            json_match = re.search(r'(\[.*\])', content, re.DOTALL)
            if json_match:
                try:
                    variations = json.loads(json_match.group(1))

                    if isinstance(variations, list) and all(isinstance(item, str) for item in variations):
                        return variations
                except json.JSONDecodeError:
                    pass

            # Fallback to regex extraction
            variations = re.findall(r'"([^"]+)"', content)
            if variations:
                return variations

            return []
        except Exception as e:
            logger.error(f"Error generating search variations: {str(e)}")
            return []

    def _search_api(self, query: str, content_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Search the IMDB API with the given query"""
        try:
            params = {"q": query}

            # Add content type filter if specified
            if content_type in ["movie", "tv"]:
                params["type"] = content_type

            logger.info(f"Searching API for: {query} (type: {content_type})")

            response = requests.get(self.api_url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                if data and "ok" in data and data["ok"] and "description" in data and data["description"]:
                    results = data["description"]

                    if results:
                        logger.info(f"Found {len(results)} results for '{query}'")

                        # Return the first result with metadata
                        return {
                            "item": results[0],
                            "type": content_type,
                            "query": query
                        }
            else:
                logger.warning(f"API returned status code {response.status_code} for '{query}'")

        except requests.RequestException as e:
            logger.error(f"Network error while searching for '{query}': {str(e)}")
        except Exception as e:
            logger.error(f"Error processing results for '{query}': {str(e)}")

        return None

    def _calculate_confidence(self, result: Dict[str, Any], media_info: MediaInfo, original_title: str) -> float:
        """Calculate confidence score for the API result"""
        if self.use_llm:
            return self._llm_validate_match(result, original_title)

        # Calculate basic confidence without LLM
        item = result["item"]
        api_title = item.get("#TITLE", "")
        api_year = item.get("#YEAR", "")

        # Start with base confidence
        confidence = 0.5

        # Add confidence for title similarity
        title_similarity = self._string_similarity(media_info.title.lower(), api_title.lower())
        similarity_factor = title_similarity * 0.3  # 0.3 weight for title similarity

        # Add confidence for year match
        year_factor = 0.3 if media_info.year and api_year == media_info.year else 0

        # Calculate total confidence
        total_confidence = confidence + similarity_factor + year_factor

        # Cap at 1.0
        return min(1.0, total_confidence)

    def _llm_validate_match(self, result: Dict[str, Any], original_title: str) -> float:
        """Use LLM to validate match"""
        if not self.use_llm or not OLLAMA_AVAILABLE:
            return 0.5

        item = result["item"]
        title = item.get("#TITLE", "")
        year = item.get("#YEAR", "")
        content_type = result.get("type", "")

        prompt = f"""
Rate how well this IMDB search result matches the original filename on a scale of 0-10.
Answer with ONLY a number from 0-10.

Original filename: {original_title}
IMDB result: {title} ({year}) - {content_type}

0 = Completely wrong match
5 = Somewhat related but not ideal
10 = Perfect match
"""
        try:
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.1, 'num_predict': 20}
            )

            content = response['message']['content'].strip()
            score_match = re.search(r'(\d+)', content)

            if score_match:
                score = int(score_match.group(1))
                if 0 <= score <= 10:
                    # Convert to 0-1 scale
                    confidence = score / 10
                    return confidence

            return 0.5  # Default to moderate confidence
        except Exception as e:
            logger.error(f"Error validating match: {str(e)}")
            return 0.5

    @staticmethod
    def _string_similarity(s1: str, s2: str) -> float:
        """Calculate string similarity using Levenshtein distance"""
        # Use difflib for better similarity computation
        return difflib.SequenceMatcher(None, s1, s2).ratio()


class MediaOrganizer:
    """Main class for organizing media files"""

    # Default folders to ignore
    DEFAULT_IGNORE_FOLDERS = [
        "System Volume Information",
        "$RECYCLE.BIN",
        "lost+found"
    ]

    def __init__(
            self,
            source_dir: str,
            movies_dir: str,
            tv_dir: str,
            api_url: str,
            use_llm: bool = False,
            dry_run: bool = True,
            ignore_folders: Optional[List[str]] = None
    ):
        self.source_dir = source_dir
        self.movies_dir = movies_dir
        self.tv_dir = tv_dir
        self.dry_run = dry_run

        # Set up folders to ignore (system folders + any user-specified)
        self.ignore_folders = set(self.DEFAULT_IGNORE_FOLDERS)
        if ignore_folders:
            self.ignore_folders.update(ignore_folders)

        # Initialize operation tracker
        self.operation_tracker = MediaOperationTracker()

        # Initialize components
        self.cleaner = MediaCleaner(use_llm=use_llm)
        self.api_client = ImdbAPIClient(api_url=api_url, use_llm=use_llm)

        # Cache of existing TV shows for faster lookup
        self.tv_series_cache = {}

    def scan_directory(self) -> None:
        """Scan the source directory and process media folders"""
        logger.info(f"Scanning source directory: {self.source_dir}")
        logger.info(f"Ignoring folders: {', '.join(self.ignore_folders)}")

        try:
            # Ensure destination directories exist
            if not self.dry_run:
                os.makedirs(self.movies_dir, exist_ok=True)
                os.makedirs(self.tv_dir, exist_ok=True)
            else:
                logger.info(f"[DRY RUN] Would create destination directories if needed")

            # Get all directories in the source
            items = os.listdir(self.source_dir)

            # Filter out ignored folders
            valid_directories = []
            for item in items:
                if item in self.ignore_folders:
                    logger.info(f"Ignoring system folder: {item}")
                    self.operation_tracker.add_ignored_folder(item)
                    continue

                full_path = os.path.join(self.source_dir, item)
                if os.path.isdir(full_path):
                    valid_directories.append(full_path)

            if not valid_directories:
                logger.info("No valid directories found in source")
                return

            # Preload existing TV series to speed up matching
            self._preload_tv_series()

            # Group related TV show folders to avoid duplicate API calls and improve organization
            related_groups = self._group_related_tv_shows(valid_directories)

            # Process directories (grouped where appropriate)
            logger.info(f"Found {len(valid_directories)} directories to process in {len(related_groups)} groups")

            for group in related_groups:
                if len(group) == 1:
                    # Single item, process normally
                    self.process_folder(group[0])
                else:
                    # Multiple related items (e.g., different seasons of same show)
                    self._process_related_folders(group)

                time.sleep(0.5)  # Small delay between groups to prevent API overload

            # Print the summary
            summary = self.operation_tracker.get_summary()
            logger.info(summary)

            if self.dry_run:
                logger.info("Dry run complete! No files were moved.")
            else:
                logger.info("Processing complete! Files have been moved.")

        except OSError as e:
            logger.error(f"Error scanning directory: {str(e)}")

    def _preload_tv_series(self) -> None:
        """Preload existing TV series to speed up matching"""
        try:
            if os.path.exists(self.tv_dir):
                for item in os.listdir(self.tv_dir):
                    item_path = os.path.join(self.tv_dir, item)
                    if os.path.isdir(item_path):
                        # Extract title and various normalized forms for matching
                        normalized_title = self._normalize_title_for_comparison(item)
                        simple_title = self._simplify_title(item)

                        # Store both forms for matching
                        self.tv_series_cache[normalized_title] = (item_path, item)
                        self.tv_series_cache[simple_title] = (item_path, item)

                logger.debug(f"Preloaded {len(self.tv_series_cache) // 2} TV series for matching")
        except OSError as e:
            logger.error(f"Error preloading TV series: {str(e)}")

    def _group_related_tv_shows(self, directories: List[str]) -> List[List[str]]:
        """Group directories that appear to be related (same TV show, different seasons)"""
        groups = []
        processed = set()

        # Sort directories to ensure consistent processing
        sorted_dirs = sorted(directories)

        for i, dir_path in enumerate(sorted_dirs):
            if dir_path in processed:
                continue

            dir_name = os.path.basename(dir_path)
            related = [dir_path]
            processed.add(dir_path)

            # Extract base title (without season info)
            base_match = re.match(r'^(.*?)(?:\s*(?:Season|S)\s*\d|\s*\(\d{4}\)\s*(?:Season|S))', dir_name,
                                  re.IGNORECASE)
            if base_match:
                base_title = base_match.group(1).strip().lower()

                # Only group if the base title is substantial (not just 1-2 chars)
                if len(base_title) > 3:
                    # Look for other directories with the same base
                    for j in range(i + 1, len(sorted_dirs)):
                        other_path = sorted_dirs[j]
                        if other_path in processed:
                            continue

                        other_name = os.path.basename(other_path)
                        if re.search(r'(?:Season|S)\s*\d', other_name, re.IGNORECASE):
                            # Check if the base titles are similar
                            if (other_name.lower().startswith(base_title) or
                                    difflib.SequenceMatcher(None, base_title,
                                                            other_name.lower()[:len(base_title)]).ratio() > 0.8):
                                related.append(other_path)
                                processed.add(other_path)

            groups.append(related)

        return groups

    def _process_related_folders(self, folders: List[str]) -> None:
        """Process multiple folders that are related (different seasons of same show)"""
        if not folders:
            return

        # Use the first folder to determine the show information
        first_folder = folders[0]
        first_name = os.path.basename(first_folder)

        logger.info(f"Processing related group: {first_name} (+{len(folders) - 1} more)")

        # Extract media info from the first folder
        media_info = self.cleaner.clean_title(first_name)

        # Enhance with API search
        api_info = self.api_client.search(media_info, first_name)

        if api_info.content_type != "tv":
            logger.warning(f"Expected TV show but detected as {api_info.content_type}, processing individually")
            for folder in folders:
                self.process_folder(folder)
            return

        # Find or create TV show folder
        existing_path, existing_name = self._find_existing_tv_series(api_info.formatted_title, first_name)

        if existing_path:
            tv_show_path = existing_path
            tv_show_name = existing_name
            logger.info(f"Found existing TV series at: {tv_show_path}")
        else:
            # Clean up the title for a new folder
            clean_title = self._clean_for_folder_name(api_info.formatted_title)
            tv_show_path = os.path.join(self.tv_dir, clean_title)
            tv_show_name = clean_title
            logger.info(f"Will create new TV series: {tv_show_path}")

        # Process each folder as a season
        for folder_path in folders:
            folder_name = os.path.basename(folder_path)

            # Extract season number
            season_match = re.search(r'(?:Season|S)\s*(\d+)', folder_name, re.IGNORECASE)
            if not season_match:
                # Check for season range
                range_match1 = re.search(MediaPattern.SEASON_RANGE_1, folder_name, re.IGNORECASE)
                range_match2 = re.search(MediaPattern.SEASON_RANGE_2, folder_name, re.IGNORECASE)

                if range_match1 or range_match2:
                    if range_match1:
                        start_season = int(range_match1.group(1))
                        end_season = int(range_match1.group(2))
                    else:
                        start_season = int(range_match2.group(1))
                        end_season = int(range_match2.group(2))

                    # Process multi-season folder
                    self._process_multi_season_folder(folder_path, tv_show_path, start_season, end_season)
                    continue
                else:
                    logger.warning(f"Could not extract season number from {folder_name}, skipping")
                    continue

            season_num = int(season_match.group(1))
            season_folder = f"Season {season_num}"
            dest_season_path = os.path.join(tv_show_path, season_folder)

            logger.info(f"Processing {folder_name} as {season_folder}")

            # Track the operation
            self.operation_tracker.add_tv_operation(folder_path, tv_show_path, season_num)

            if not self.dry_run:
                try:
                    # Create TV show and season directories
                    os.makedirs(tv_show_path, exist_ok=True)

                    # Check if season folder already exists
                    if os.path.exists(dest_season_path):
                        logger.warning(f"Season folder already exists: {dest_season_path}")
                        self.operation_tracker.add_error(folder_path, f"Season folder already exists: {season_folder}")
                        continue

                    # Create season folder
                    os.makedirs(dest_season_path, exist_ok=True)

                    # Move all content to the season folder
                    logger.info(f"Moving content to: {dest_season_path}")
                    for item in os.listdir(folder_path):
                        item_path = os.path.join(folder_path, item)
                        dest_item_path = os.path.join(dest_season_path, item)
                        shutil.move(item_path, dest_item_path)

                    # Remove empty source folder
                    if not os.listdir(folder_path):
                        os.rmdir(folder_path)

                    logger.info(f"Successfully moved content to: {dest_season_path}")
                except (OSError, shutil.Error) as e:
                    logger.error(f"Error moving content: {str(e)}")
                    self.operation_tracker.add_error(folder_path, f"Error moving content: {str(e)}")
            else:
                logger.info(f"[DRY RUN] Would create: {dest_season_path}")
                logger.info(f"[DRY RUN] Would move content from {folder_path} to {dest_season_path}")

    def process_folder(self, folder_path: str) -> None:
        """Process a single folder"""
        folder_name = os.path.basename(folder_path)
        logger.info(f"Processing: {folder_name}")

        # Step 1: Clean the title and detect media type
        media_info = self.cleaner.clean_title(folder_name)
        logger.info(f"Initial extraction: {media_info.formatted_title} ({media_info.content_type})")

        # Step 2: Search API for better metadata
        api_info = self.api_client.search(media_info, folder_name)

        # Step 3: Process based on content type
        if api_info.content_type == "movie":
            self.process_movie(folder_path, api_info)
        else:
            # Check for season range
            if api_info.season_range:
                start_season, end_season = api_info.season_range
                # Find or create TV show folder
                existing_path, existing_name = self._find_existing_tv_series(api_info.formatted_title, folder_name)

                if existing_path:
                    tv_show_path = existing_path
                    logger.info(f"Found existing TV series at: {tv_show_path}")
                else:
                    # Clean up the title for a new folder
                    clean_title = self._clean_for_folder_name(api_info.formatted_title)
                    tv_show_path = os.path.join(self.tv_dir, clean_title)
                    logger.info(f"No existing TV series found. Will create: {tv_show_path}")

                # Process multi-season folder
                self._process_multi_season_folder(folder_path, tv_show_path, start_season, end_season)
            else:
                self.process_tv_show(folder_path, api_info)

    def process_movie(self, folder_path: str, media_info: MediaInfo) -> None:
        """Process a movie folder"""
        # Clean the movie title even more for destination
        clean_movie_title = self._clean_for_folder_name(media_info.formatted_title)
        dest_path = os.path.join(self.movies_dir, clean_movie_title)

        # Check if destination already exists
        if os.path.exists(dest_path):
            logger.warning(f"Destination already exists: {dest_path}")
            self.operation_tracker.add_error(folder_path, "Destination already exists")
            return

        # Move the directory
        logger.info(f"Moving movie to: {dest_path}")
        self.operation_tracker.add_movie_operation(folder_path, dest_path)

        if not self.dry_run:
            try:
                shutil.move(folder_path, dest_path)
                logger.info(f"Successfully moved to: {dest_path}")
            except (OSError, shutil.Error) as e:
                logger.error(f"Error moving: {str(e)}")
                self.operation_tracker.add_error(folder_path, f"Error moving: {str(e)}")
        else:
            logger.info(f"[DRY RUN] Would move '{folder_path}' to '{dest_path}'")

    def process_tv_show(self, folder_path: str, media_info: MediaInfo) -> None:
        """Process a TV show folder"""
        # Find existing TV series or create new path
        existing_path, existing_name = self._find_existing_tv_series(media_info.formatted_title,
                                                                     os.path.basename(folder_path))

        # Determine which title to use (prefer existing name if found)
        if existing_path:
            tv_show_path = existing_path
            tv_show_name = existing_name
            logger.info(f"Found existing TV series at: {tv_show_path}")
        else:
            # Clean up the title for a new folder
            clean_title = self._clean_for_folder_name(media_info.formatted_title)
            tv_show_path = os.path.join(self.tv_dir, clean_title)
            tv_show_name = clean_title
            logger.info(f"No existing TV series found. Will create: {tv_show_path}")

        # Handle based on whether we have a season number
        if media_info.season:
            # Format the season folder name
            season_folder = f"Season {media_info.season}"
            dest_season_path = os.path.join(tv_show_path, season_folder)

            # Track the operation
            self.operation_tracker.add_tv_operation(folder_path, tv_show_path, media_info.season)

            if not self.dry_run:
                try:
                    # Create TV show and season directories
                    os.makedirs(tv_show_path, exist_ok=True)

                    # Check if season folder already exists
                    if os.path.exists(dest_season_path):
                        logger.warning(f"Season folder already exists: {dest_season_path}")
                        self.operation_tracker.add_error(folder_path, f"Season folder already exists: {season_folder}")
                        return

                    # Create season folder
                    os.makedirs(dest_season_path, exist_ok=True)

                    # Move all content to the season folder
                    logger.info(f"Moving content to: {dest_season_path}")
                    for item in os.listdir(folder_path):
                        item_path = os.path.join(folder_path, item)
                        dest_item_path = os.path.join(dest_season_path, item)
                        shutil.move(item_path, dest_item_path)

                    # Remove empty source folder
                    if not os.listdir(folder_path):
                        os.rmdir(folder_path)

                    logger.info(f"Successfully moved content to: {dest_season_path}")
                except (OSError, shutil.Error) as e:
                    logger.error(f"Error moving content: {str(e)}")
                    self.operation_tracker.add_error(folder_path, f"Error moving content: {str(e)}")
            else:
                logger.info(f"[DRY RUN] Would create: {dest_season_path}")
                logger.info(f"[DRY RUN] Would move content from {folder_path} to {dest_season_path}")
        else:
            # Check if this is likely a multi-season pack (should be handled by the caller)
            folder_name = os.path.basename(folder_path)
            season_range_match = re.search(r'Season\s*(\d+)[- ](\d+)', folder_name, re.IGNORECASE)
            s_range_match = re.search(r'S(\d{1,2})[-_]S(\d{1,2})', folder_name, re.IGNORECASE)

            # If we found a season range, handle it specially
            if season_range_match or s_range_match:
                if season_range_match:
                    start_season = int(season_range_match.group(1))
                    end_season = int(season_range_match.group(2))
                else:
                    start_season = int(s_range_match.group(1))
                    end_season = int(s_range_match.group(2))

                # Process multi-season folder
                self._process_multi_season_folder(folder_path, tv_show_path, start_season, end_season)
            else:
                # Check if there's a season indicator in the title
                # For titles like "Twisted Metal - Season 1 (2023)" which should be treated as a specific season
                season_indicator_match = re.search(r'(?:Season|S)\s*(\d+)', folder_name, re.IGNORECASE)
                if season_indicator_match:
                    # Extract the season number
                    season_num = int(season_indicator_match.group(1))

                    # Format the season folder name
                    season_folder = f"Season {season_num}"
                    dest_season_path = os.path.join(tv_show_path, season_folder)

                    # Track the operation
                    self.operation_tracker.add_tv_operation(folder_path, tv_show_path, season_num)

                    if not self.dry_run:
                        try:
                            # Create TV show and season directories
                            os.makedirs(tv_show_path, exist_ok=True)

                            # Check if season folder already exists
                            if os.path.exists(dest_season_path):
                                logger.warning(f"Season folder already exists: {dest_season_path}")
                                self.operation_tracker.add_error(folder_path,
                                                                 f"Season folder already exists: {season_folder}")
                                return

                            # Create season folder
                            os.makedirs(dest_season_path, exist_ok=True)

                            # Move all content to the season folder
                            logger.info(f"Season indicator in title. Moving content to: {dest_season_path}")
                            for item in os.listdir(folder_path):
                                item_path = os.path.join(folder_path, item)
                                dest_item_path = os.path.join(dest_season_path, item)
                                shutil.move(item_path, dest_item_path)

                            # Remove empty source folder
                            if not os.listdir(folder_path):
                                os.rmdir(folder_path)

                            logger.info(f"Successfully moved content to: {dest_season_path}")
                        except (OSError, shutil.Error) as e:
                            logger.error(f"Error moving content: {str(e)}")
                            self.operation_tracker.add_error(folder_path, f"Error moving content: {str(e)}")
                    else:
                        logger.info(f"Season indicator in title. [DRY RUN] Would create: {dest_season_path}")
                        logger.info(f"[DRY RUN] Would move content from {folder_path} to {dest_season_path}")
                else:
                    # Regular single-season or complete series
                    logger.info(f"No season detected, treating as complete series")

                    # Track the operation
                    self.operation_tracker.add_tv_operation(folder_path, tv_show_path, None)

                    # Move the entire folder
                    dest_path = os.path.join(self.tv_dir, tv_show_name)

                    if os.path.exists(dest_path):
                        logger.warning(f"Destination already exists: {dest_path}")
                        self.operation_tracker.add_error(folder_path, "Destination already exists")
                        return

                    if not self.dry_run:
                        try:
                            shutil.move(folder_path, dest_path)
                            logger.info(f"Successfully moved to: {dest_path}")
                        except (OSError, shutil.Error) as e:
                            logger.error(f"Error moving: {str(e)}")
                            self.operation_tracker.add_error(folder_path, f"Error moving: {str(e)}")
                    else:
                        logger.info(f"[DRY RUN] Would move '{folder_path}' to '{dest_path}'")

    def _process_multi_season_folder(self, folder_path: str, tv_show_path: str, start_season: int,
                                     end_season: int) -> None:
        """Process a folder containing multiple seasons"""
        logger.info(f"Detected multi-season pack from Season {start_season} to {end_season}")

        # Track the main operation
        self.operation_tracker.add_tv_operation(folder_path, tv_show_path, None)

        # For multi-season content, we'll check for a season structure in the folder
        season_folders = []

        try:
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)
                if os.path.isdir(item_path):
                    # Check if it's a season folder
                    season_match = re.search(r'(?:Season|S)\s*(\d+)', item, re.IGNORECASE)
                    if season_match:
                        season_num = int(season_match.group(1))
                        season_folders.append((item_path, season_num))
        except OSError as e:
            logger.error(f"Error checking for season folders: {str(e)}")
            self.operation_tracker.add_error(folder_path, f"Error checking for season folders: {str(e)}")

        if season_folders:
            # If we found season folders inside, process each one
            logger.info(f"Found {len(season_folders)} season folders inside multi-season pack")

            if not self.dry_run:
                try:
                    # Create TV show directory
                    os.makedirs(tv_show_path, exist_ok=True)

                    # Process each season folder
                    for season_path, season_num in season_folders:
                        dest_season_path = os.path.join(tv_show_path, f"Season {season_num}")

                        # Check if destination already exists
                        if os.path.exists(dest_season_path):
                            logger.warning(f"Season folder already exists: {dest_season_path}")
                            self.operation_tracker.add_error(season_path,
                                                             f"Season folder already exists: Season {season_num}")
                            continue

                        # Create and move
                        os.makedirs(dest_season_path, exist_ok=True)
                        logger.info(f"Moving season {season_num} to: {dest_season_path}")

                        # Move content
                        for item in os.listdir(season_path):
                            item_path = os.path.join(season_path, item)
                            dest_item_path = os.path.join(dest_season_path, item)
                            shutil.move(item_path, dest_item_path)

                    # Remove empty source folders
                    for season_path, _ in season_folders:
                        if os.path.exists(season_path) and not os.listdir(season_path):
                            os.rmdir(season_path)

                    if os.path.exists(folder_path) and not os.listdir(folder_path):
                        os.rmdir(folder_path)

                except (OSError, shutil.Error) as e:
                    logger.error(f"Error processing multi-season pack: {str(e)}")
                    self.operation_tracker.add_error(folder_path, f"Error processing multi-season pack: {str(e)}")
            else:
                for season_path, season_num in season_folders:
                    dest_season_path = os.path.join(tv_show_path, f"Season {season_num}")
                    logger.info(f"[DRY RUN] Would create: {dest_season_path}")
                    logger.info(f"[DRY RUN] Would move Season {season_num} content to {dest_season_path}")
        else:
            # No season folders inside, create separate season folders for each season in the range
            if not self.dry_run:
                try:
                    # Create TV show directory
                    os.makedirs(tv_show_path, exist_ok=True)

                    # Create all season folders in the range
                    for season_num in range(start_season, end_season + 1):
                        season_path = os.path.join(tv_show_path, f"Season {season_num}")
                        os.makedirs(season_path, exist_ok=True)
                        logger.info(f"Created season folder: {season_path}")

                    # Look for individual season content inside main folder
                    season_files = {}
                    for item in os.listdir(folder_path):
                        item_path = os.path.join(folder_path, item)
                        if os.path.isfile(item_path):
                            # Try to determine which season this file belongs to
                            item_season = None
                            for s in range(start_season, end_season + 1):
                                if re.search(rf'[sS]0*{s}[eE]\d+', item, re.IGNORECASE):
                                    item_season = s
                                    break

                            # If we found a season, add to that season's files
                            if item_season:
                                if item_season not in season_files:
                                    season_files[item_season] = []
                                season_files[item_season].append(item_path)
                            else:
                                # If no specific season, add to first season
                                if start_season not in season_files:
                                    season_files[start_season] = []
                                season_files[start_season].append(item_path)

                    # Move files to appropriate season folders
                    for season_num, files in season_files.items():
                        season_path = os.path.join(tv_show_path, f"Season {season_num}")
                        logger.info(f"Moving {len(files)} files to Season {season_num}")
                        for file_path in files:
                            dest_path = os.path.join(season_path, os.path.basename(file_path))
                            shutil.move(file_path, dest_path)

                    # Check if there are any files that weren't processed
                    remaining_files = [f for f in os.listdir(folder_path) if
                                       os.path.isfile(os.path.join(folder_path, f))]
                    if remaining_files:
                        # Move remaining files to the first season
                        first_season_path = os.path.join(tv_show_path, f"Season {start_season}")
                        logger.info(f"Moving {len(remaining_files)} remaining files to Season {start_season}")
                        for file_name in remaining_files:
                            file_path = os.path.join(folder_path, file_name)
                            dest_path = os.path.join(first_season_path, file_name)
                            shutil.move(file_path, dest_path)

                    # Clean up empty source folder
                    if os.path.exists(folder_path) and len(os.listdir(folder_path)) == 0:
                        os.rmdir(folder_path)

                except (OSError, shutil.Error) as e:
                    logger.error(f"Error creating season folders: {str(e)}")
                    self.operation_tracker.add_error(folder_path, f"Error creating season folders: {str(e)}")
            else:
                # Dry run output
                for season_num in range(start_season, end_season + 1):
                    season_path = os.path.join(tv_show_path, f"Season {season_num}")
                    logger.info(f"[DRY RUN] Would create: {season_path}")

                logger.info(f"[DRY RUN] Would move files to appropriate season folders")
                logger.info(
                    f"[DRY RUN] Would move remaining files to: {os.path.join(tv_show_path, f'Season {start_season}')}")

    def _clean_for_folder_name(self, title: str) -> str:
        """Clean a title to make it suitable for a folder name"""
        # Remove extra spaces
        clean = re.sub(r'\s+', ' ', title).strip()

        # Remove technical specs that might still be in the title
        clean = re.sub(MediaPattern.TECHNICAL, '', clean, flags=re.IGNORECASE).strip()

        # Remove any of the noise words
        for noise_word in MediaPattern.TITLE_NOISE_WORDS:
            # Only remove if it's a complete word
            clean = re.sub(r'\b' + re.escape(noise_word) + r'\b', '', clean, flags=re.IGNORECASE)

        # Remove ( ) and [ ] that might be empty or contain minimal content
        clean = re.sub(r'\(\s*\)', '', clean)
        clean = re.sub(r'\[\s*\]', '', clean)

        # Remove trailing spaces before parentheses
        clean = re.sub(r'\s+\(', ' (', clean)

        # Clean up multiple spaces
        clean = re.sub(r'\s+', ' ', clean).strip()

        # If we still have a year in parentheses at the end, keep it, otherwise clean up
        year_match = re.search(r'\((\d{4})\)$', clean)
        if not year_match:
            # Remove trailing parenthetical expressions
            clean = re.sub(r'\s*\([^\)]*\)$', '', clean)

        # Final cleanup
        clean = re.sub(r'\s+', ' ', clean).strip()

        return clean.strip()

    def _find_existing_tv_series(self, formatted_title: str, original_filename: str) -> Optional[Tuple[str, str]]:
        """Look for existing TV series in the TV directory using improved matching

        Returns:
            Tuple of (path, folder_name) if found, None otherwise
        """
        # First try normalized title matching using the cache
        normalized_title = self._normalize_title_for_comparison(formatted_title)
        simple_title = self._simplify_title(formatted_title)

        # Check the cache for a direct match
        if normalized_title in self.tv_series_cache:
            return self.tv_series_cache[normalized_title]

        # Check for simple title match
        if simple_title in self.tv_series_cache:
            return self.tv_series_cache[simple_title]

        # If no match, try a more complex search
        # Extract the title without year for flexible matching
        base_title = re.sub(r'\s*\(\d{4}\)$', '', formatted_title).strip().lower()

        # Clean the base title further for better matching (remove common words)
        clean_base = re.sub(r'^(the|a|an)\s+', '', base_title)

        # Also prepare a very short version for matching (first few words)
        short_base = ' '.join(base_title.split()[:2]) if len(base_title.split()) > 2 else base_title

        # Base title without technical specs
        base_cleaned = self._deep_clean_title_for_comparison(base_title)

        logger.debug(f"Looking for TV series match: '{base_title}', '{clean_base}', '{short_base}', '{base_cleaned}'")

        try:
            best_match = None
            best_score = 0.0

            for item in os.listdir(self.tv_dir):
                item_path = os.path.join(self.tv_dir, item)

                if os.path.isdir(item_path):
                    # Clean up the item name for comparison
                    item_base = re.sub(r'\s*\(\d{4}\)$', '', item).strip().lower()
                    clean_item = re.sub(r'^(the|a|an)\s+', '', item_base)
                    short_item = ' '.join(item_base.split()[:2]) if len(item_base.split()) > 2 else item_base
                    item_cleaned = self._deep_clean_title_for_comparison(item_base)

                    # Multiple matching strategies with progressive scores
                    score = 0.0

                    # 1. Exact match is best
                    if item_base == base_title:
                        score = 1.0
                    # 2. Clean match is very good
                    elif clean_item == clean_base:
                        score = 0.95
                    # 3. Simplified title match is good
                    elif self._simplify_title(item) == self._simplify_title(formatted_title):
                        score = 0.9
                    # 4. Normalized title match is good
                    elif self._normalize_title_for_comparison(item) == normalized_title:
                        score = 0.85
                    # 5. Deep cleaned title match
                    elif item_cleaned == base_cleaned:
                        score = 0.8
                    else:
                        # 6. For very distinctive titles, a match on first 1-2 words is often enough
                        if len(short_base) > 5 and short_item == short_base:
                            score = 0.75
                        # 7. Use string similarity as a fallback
                        else:
                            # Check direct similarity
                            direct_similarity = difflib.SequenceMatcher(None, item_base, base_title).ratio()

                            # Also check similarity of cleaned titles
                            clean_similarity = difflib.SequenceMatcher(None, item_cleaned, base_cleaned).ratio()

                            # Use the better of the two similarities
                            similarity = max(direct_similarity, clean_similarity)

                            if similarity > 0.85:  # Higher threshold for safety
                                score = 0.7 * similarity  # Scale down a bit as it's less certain

                    # Track the best match
                    if score > best_score:
                        best_score = score
                        best_match = (item_path, item)

            # If we found a good match, return it
            if best_match and best_score > 0.7:
                logger.info(
                    f"Found matching TV series: '{best_match[1]}' for '{formatted_title}' (score: {best_score:.2f})")

                # Add to cache for future lookups
                self.tv_series_cache[normalized_title] = best_match
                self.tv_series_cache[simple_title] = best_match

                return best_match

        except OSError as e:
            logger.error(f"Error finding existing TV series: {str(e)}")

        return None, None

    def _normalize_title_for_comparison(self, title: str) -> str:
        """Create a normalized version of the title for comparison"""
        # Extract base title without year
        if '(' in title and ')' in title and re.search(r'\(\d{4}\)', title):
            base = re.sub(r'\s*\(\d{4}\).*$', '', title)
        else:
            base = title

        # Convert to lowercase
        normalized = base.lower()

        # Remove articles
        normalized = re.sub(r'^(the|a|an)\s+', '', normalized)

        # Remove punctuation and replace with spaces
        normalized = re.sub(r'[^\w\s]', ' ', normalized)

        # Remove multiple spaces
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        # Remove technical terms and noise words
        normalized = re.sub(MediaPattern.TECHNICAL, '', normalized, flags=re.IGNORECASE).strip()

        for noise_word in MediaPattern.TITLE_NOISE_WORDS:
            normalized = re.sub(r'\b' + re.escape(noise_word) + r'\b', '', normalized, flags=re.IGNORECASE)

        # Remove duplicate spaces
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        return normalized

    def _simplify_title(self, title: str) -> str:
        """Create a simplified version of the title (just main words)"""
        # Extract base title without year
        if '(' in title and ')' in title and re.search(r'\(\d{4}\)', title):
            base = re.sub(r'\s*\(\d{4}\).*$', '', title)
        else:
            base = title

        # Convert to lowercase and remove punctuation
        simplified = re.sub(r'[^\w\s]', ' ', base.lower())

        # Remove articles and common words
        simplified = re.sub(r'^(the|a|an)\s+', '', simplified)

        # Remove duplicate spaces
        simplified = re.sub(r'\s+', ' ', simplified).strip()

        # Take just the first 2-3 substantial words if the title is long
        words = simplified.split()
        if len(words) > 3:
            # Use words that are at least 3 characters (skip short words)
            substantial_words = [w for w in words if len(w) >= 3]
            if len(substantial_words) >= 2:
                return ' '.join(substantial_words[:2])

        return simplified

    def _deep_clean_title_for_comparison(self, title: str) -> str:
        """Deep clean a title for better comparison"""
        # Convert to lowercase
        cleaned = title.lower()

        # Remove all punctuation and special characters
        cleaned = re.sub(r'[^\w\s]', ' ', cleaned)

        # Remove articles
        cleaned = re.sub(r'^(the|a|an)\s+', '', cleaned)

        # Remove technical terms and specifications
        cleaned = re.sub(MediaPattern.TECHNICAL, '', cleaned, flags=re.IGNORECASE)

        # Remove common noise words
        for noise_word in MediaPattern.TITLE_NOISE_WORDS:
            cleaned = re.sub(r'\b' + re.escape(noise_word) + r'\b', '', cleaned, flags=re.IGNORECASE)

        # Remove multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        return cleaned

    @staticmethod
    def _string_similarity(s1: str, s2: str) -> float:
        """Calculate string similarity"""
        # Use difflib for better similarity computation
        return difflib.SequenceMatcher(None, s1, s2).ratio()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Organize media files into Movies and TV Show directories')
    parser.add_argument('--no-llm', action='store_true', help='Disable LLM features')
    parser.add_argument('--dry-run', action='store_true', help='Simulate without moving files')
    parser.add_argument('--real', action='store_true', help='Actually move files (not dry run)')
    parser.add_argument('--source', type=str, help='Source directory', default=MAIN_DIR)
    parser.add_argument('--movies', type=str, help='Movies directory', default=MOVIES_DIR)
    parser.add_argument('--tv', type=str, help='TV Shows directory', default=TV_DIR)
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--ignore', type=str, nargs='+', help='Additional folders to ignore')

    return parser.parse_args()


def main() -> None:
    """Main entry point"""
    args = parse_args()

    # Configure logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Determine whether to use LLM
    use_llm = not args.no_llm and OLLAMA_AVAILABLE

    # Determine whether to actually move files
    dry_run = not args.real if args.real else args.dry_run if args.dry_run else DRY_RUN

    logger.info("Starting media organizer")
    logger.info(f"LLM features: {'Enabled' if use_llm else 'Disabled'}")
    logger.info(f"Mode: {'Dry run (no files will be moved)' if dry_run else 'Move files'}")

    # Create and run the organizer
    organizer = MediaOrganizer(
        source_dir=args.source,
        movies_dir=args.movies,
        tv_dir=args.tv,
        api_url=API_URL,
        use_llm=use_llm,
        dry_run=dry_run,
        ignore_folders=args.ignore
    )

    organizer.scan_directory()


if __name__ == "__main__":
    main()