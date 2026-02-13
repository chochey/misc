# Media Renamer Configuration

# Directories
SOURCE_DIR = r"/media/blue/ST6000NM 0115-1YZ110 USB Device/Share"
INCOMPLETE_DIR = r"/media/blue/ST6000NM 0115-1YZ110 USB Device/Share/incomplete"

# Primary destination (default for new content with no existing folder)
MOVIE_DEST_DIR = r"/media/blue/OOS6000G USB Device/Movies"
TV_DEST_DIR = r"/media/blue/OOS6000G USB Device/TV"

# All destination drives (searched for existing content, used by --library/--library-tv)
# To add a new drive, just append another path to each list
MOVIE_DEST_DIRS = [
    r"/media/blue/OOS6000G USB Device/Movies",
    r"/media/blue/New Volume/Movies",
]
TV_DEST_DIRS = [
    r"/media/blue/OOS6000G USB Device/TV",
    r"/media/blue/New Volume/TV",
]

# OMDb API
OMDB_API_KEY = "4882f1b4"
OMDB_BASE_URL = "http://www.omdbapi.com/"

# Watch mode settings
POLL_INTERVAL = 30  # seconds

# Library progress tracking (stores which folders have been processed)
LIBRARY_PROGRESS_FILE = "library_progress.json"

# Supported file extensions
VIDEO_EXTENSIONS = [".mkv", ".mp4", ".avi", ".m4v", ".webm", ".mov"]
SUBTITLE_EXTENSIONS = [".srt", ".sub", ".ass", ".idx", ".ssa", ".vtt"]
