# Media Renamer Configuration

# Directories (using mergerfs merged mount at /mnt/media)
SOURCE_DIR = r"/mnt/media/Share"
INCOMPLETE_DIR = r"/mnt/media/incomplete"

# Primary destination (default for new content with no existing folder)
MOVIE_DEST_DIR = r"/mnt/media/Movies"
TV_DEST_DIR = r"/mnt/media/TV"

# All destination drives (searched for existing content, used by --library/--library-tv)
# With mergerfs, all drives are combined so only one path is needed per type
MOVIE_DEST_DIRS = [
    r"/mnt/media/Movies",
]
TV_DEST_DIRS = [
    r"/mnt/media/TV",
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
