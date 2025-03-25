#!/usr/bin/env python3
import os
import sys
import time
import logging
# Removed argparse import as we're using hardcoded config
import tempfile
import subprocess
import threading
from datetime import datetime

# Constants
DEFAULT_SCAN_INTERVAL = 300  # 5 minutes
DEFAULT_STABLE_TIME = 60  # 1 minute
DEFAULT_MAX_WALK_TIME = 300  # 5 minutes
DEFAULT_SCRIPT_TIMEOUT = 1800  # 30 minutes
DEFAULT_COOLDOWN = 3600  # 1 hour

# Global settings
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
SHUTDOWN_FLAG = threading.Event()

# Set up logger
logger = logging.getLogger('MediaMonitor')
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
logger.addHandler(console_handler)


# Note: File logging is intentionally disabled to avoid file creation

def validate_environment():
    """Validate that the environment has necessary utilities."""
    try:
        # Add any environment checks here
        return True
    except Exception as e:
        logger.error(f"Environment validation failed: {e}")
        return False


def has_valid_permissions(path):
    """Check if we have read and execute permissions for the path."""
    if not os.path.exists(path):
        return False

    try:
        # Check read access
        if os.path.isdir(path):
            os.listdir(path)
        else:
            with open(path, 'r'):
                pass
        return True
    except (PermissionError, OSError):
        return False


def normalize_path(path):
    """Safely normalize a path and check permissions."""
    try:
        if not path:
            return None
        normalized = os.path.normpath(os.path.abspath(os.path.expanduser(path)))
        if not os.path.exists(normalized):
            logger.warning(f"Path does not exist: {normalized}")
            return None
        if not has_valid_permissions(normalized):
            logger.warning(f"Insufficient permissions for path: {normalized}")
            return None
        return normalized
    except Exception as e:
        logger.warning(f"Error normalizing path {path}: {e}")
        return None


def get_total_size(path, max_time=DEFAULT_MAX_WALK_TIME):
    """
    Get the total size of a directory or file, avoiding symlinks.
    Limits time spent to max_time seconds.
    """
    if not os.path.exists(path):
        return 0

    if os.path.isfile(path):
        try:
            return os.path.getsize(path)
        except (PermissionError, OSError) as e:
            logger.debug(f"Cannot get size of file {path}: {e}")
            return 0

    start_time = time.time()
    total_size = 0

    try:
        for dirpath, dirnames, filenames in os.walk(path, followlinks=False):  # Avoid following symlinks
            if time.time() - start_time > max_time or SHUTDOWN_FLAG.is_set():
                logger.debug(f"Size calculation for {path} timed out or interrupted")
                break

            # Filter directories without proper permissions
            dirnames[:] = [d for d in dirnames if has_valid_permissions(os.path.join(dirpath, d))]

            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    if os.path.islink(fp):
                        continue  # Skip symbolic links
                    total_size += os.path.getsize(fp)
                except Exception as e:
                    logger.debug(f"Skipping file {fp} due to error: {e}")
    except Exception as e:
        logger.warning(f"Error walking directory {path}: {e}")

    return total_size


class DirectoryMonitor:
    """Monitor directories for stability."""

    def __init__(self, base_dir, stable_time=DEFAULT_STABLE_TIME):
        self.base_dir = base_dir
        self.stable_time = stable_time
        self.dir_states = {}  # Dictionary to track directory state

    def get_stable_directories(self):
        """Identify directories that have been stable for the specified time period."""
        current_time = time.time()
        stable_dirs = []

        try:
            # Scan the base directory for subdirectories
            subdirs = []
            base_dir = normalize_path(self.base_dir)
            if not base_dir:
                logger.error(f"Base directory is invalid: {self.base_dir}")
                return []

            try:
                subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir)
                           if os.path.isdir(os.path.join(base_dir, d))]
            except (PermissionError, OSError) as e:
                logger.error(f"Cannot access base directory: {e}")
                return []

            # Get current status for each directory
            new_states = {}
            for directory in subdirs:
                directory = normalize_path(directory)
                if directory is None:
                    continue

                try:
                    size = get_total_size(directory)
                    mtime = os.path.getmtime(directory)

                    if directory in self.dir_states:
                        prev_size, prev_mtime, first_seen = self.dir_states[directory]

                        # Directory is stable if:
                        # 1. Size hasn't changed
                        # 2. Modification time hasn't changed
                        # 3. It's been observed for at least stable_time
                        if (size == prev_size and
                                mtime == prev_mtime and
                                (current_time - first_seen) >= self.stable_time):
                            stable_dirs.append(directory)

                        new_states[directory] = (size, mtime, first_seen)
                    else:
                        # First time seeing this directory
                        new_states[directory] = (size, mtime, current_time)
                except Exception as e:
                    logger.warning(f"Error checking stability for {directory}: {e}")

            # Update state
            self.dir_states = new_states

        except Exception as e:
            logger.error(f"Error in get_stable_directories: {e}")

        return stable_dirs


class TransferHandler:
    """Handle the processing of stable directories."""

    def __init__(self, script_path, script_timeout=DEFAULT_SCRIPT_TIMEOUT, cooldown=DEFAULT_COOLDOWN):
        self.script_path = script_path
        self.script_timeout = script_timeout
        self.cooldown = cooldown
        self.processed_dirs = {}  # Dictionary to track processed directories

    def should_process_directory(self, directory):
        """Check if a directory should be processed based on cooldown period."""
        current_time = datetime.now()
        if directory in self.processed_dirs:
            elapsed = (current_time - self.processed_dirs[directory]).total_seconds()
            if elapsed < self.cooldown:
                logger.debug(f"Directory {directory} is in cooldown period ({elapsed:.1f}s elapsed)")
                return False
        return True

    def process_directory(self, directory):
        """Process a stable directory using the specified script."""
        if not self.should_process_directory(directory):
            return

        logger.info(f"Processing directory: {directory}")

        try:
            # Check if script is a Python file
            is_python = self.script_path.lower().endswith('.py')

            if is_python:
                # Use the Python interpreter from your virtual environment
                venv_python = r"C:\Users\blue\PycharmProjects\discord-bot\.venv\Scripts\python.exe"
                # Use --main-dir argument instead of positional argument
                cmd = [venv_python, self.script_path, "--main-dir", directory]
            else:
                cmd = [self.script_path, directory]

            logger.debug(f"Running command: {' '.join(cmd)}")

            # Run the process
            try:
                result = subprocess.run(
                    cmd,
                    timeout=self.script_timeout,
                    check=True,
                    capture_output=True,
                    text=True
                )
                logger.info(f"Script completed successfully for {directory}")
                logger.debug(f"Script output: {result.stdout}")

                # Mark as processed
                self.processed_dirs[directory] = datetime.now()

            except subprocess.TimeoutExpired:
                logger.error(f"Script timeout reached for {directory}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Script exited with code {e.returncode} for {directory}")
                logger.debug(f"Error output: {e.stderr}")
            except Exception as e:
                logger.error(f"Unexpected error running script: {e}")
        except Exception as e:
            logger.error(f"Error in process_directory: {e}")


def acquire_single_instance_lock(timeout=10):
    """Acquire a lock file to ensure only one instance of the script runs."""
    lock_file = os.path.join(tempfile.gettempdir(), 'media_monitor.lock')

    # Try to acquire the lock
    start_time = time.time()
    while os.path.exists(lock_file) and time.time() - start_time < timeout:
        try:
            # Check if the process is still running
            with open(lock_file, 'r') as f:
                pid = int(f.read().strip())
                try:
                    # Check if process exists
                    os.kill(pid, 0)
                    # Process exists, wait and retry
                    time.sleep(0.5)
                    continue
                except OSError:
                    # Process doesn't exist, we can take the lock
                    logger.warning(f"Removing stale lock file from dead process {pid}")
                    os.remove(lock_file)
                    break
        except (ValueError, IOError) as e:
            logger.warning(f"Error checking lock file: {e}")
            # Lock file is corrupt or unreadable, try to remove it
            try:
                os.remove(lock_file)
                break
            except OSError:
                pass
        time.sleep(0.5)

    if os.path.exists(lock_file):
        logger.error("Another instance is already running.")
        sys.exit(1)

    try:
        with open(lock_file, 'w') as f:
            f.write(str(os.getpid()))
    except Exception as e:
        logger.error(f"Failed to create lock file: {e}")
        sys.exit(1)

    return lock_file


def release_single_instance_lock(lock_file):
    """Release the lock file."""
    if os.path.exists(lock_file):
        try:
            os.remove(lock_file)
        except OSError as e:
            logger.warning(f"Failed to remove lock file: {e}")


def signal_handler(signum, frame):
    """Handle termination signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    SHUTDOWN_FLAG.set()


# Configuration
class Config:
    """Configuration settings."""
    # Required settings
    MONITOR_DIR = r"I:\share-downloads\complete"  # Directory to monitor
    SCRIPT_PATH = r"C:\Users\blue\PycharmProjects\misc\Python Scripts\media_organizer\test.py"  # Script to run

    # Optional settings with defaults
    SCAN_INTERVAL = 30  # Scan interval in seconds (changed from 300 to 30)
    STABLE_TIME = 30  # Time in seconds a directory must be stable (changed from 60 to 30)
    SCRIPT_TIMEOUT = DEFAULT_SCRIPT_TIMEOUT  # Maximum time in seconds for script execution
    COOLDOWN = DEFAULT_COOLDOWN  # Cooldown period in seconds

    # Logging settings
    VERBOSE = True  # Enable more detailed logging (changed from False to True)


def main():
    """Main function."""
    if not validate_environment():
        return 1

    # Set log level
    if Config.VERBOSE:
        logger.setLevel(logging.DEBUG)

    # Normalize paths
    monitor_dir = normalize_path(Config.MONITOR_DIR)
    if not monitor_dir:
        logger.error(f"Invalid monitor directory: {Config.MONITOR_DIR}")
        return 1

    script_path = normalize_path(Config.SCRIPT_PATH)
    if not script_path:
        logger.error(f"Invalid script path: {Config.SCRIPT_PATH}")
        return 1

    # Acquire lock
    lock_file = acquire_single_instance_lock()

    try:
        # Set up signal handlers
        import signal
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        logger.info(f"Starting media monitor for directory: {monitor_dir}")
        logger.info(f"Using script: {script_path}")
        if script_path.lower().endswith('.py'):
            logger.info("Python script detected, will use Python interpreter")
        logger.info(f"Scan interval: {Config.SCAN_INTERVAL} seconds")
        logger.info(f"Stability threshold: {Config.STABLE_TIME} seconds")

        # Create monitor and handler
        monitor = DirectoryMonitor(monitor_dir, Config.STABLE_TIME)
        handler = TransferHandler(script_path, Config.SCRIPT_TIMEOUT, Config.COOLDOWN)

        # Main loop
        while not SHUTDOWN_FLAG.is_set():
            try:
                logger.debug("Scanning for stable directories...")
                stable_dirs = monitor.get_stable_directories()

                if stable_dirs:
                    logger.info(f"Found {len(stable_dirs)} stable directories")
                    for directory in stable_dirs:
                        handler.process_directory(directory)

                # Wait for next scan, but be responsive to shutdown signals
                for _ in range(Config.SCAN_INTERVAL):
                    if SHUTDOWN_FLAG.is_set():
                        break
                    time.sleep(1)

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(10)  # Avoid thrashing if there's a persistent error

        logger.info("Shutdown signal received, exiting...")
        return 0

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

    finally:
        release_single_instance_lock(lock_file)


if __name__ == "__main__":
    sys.exit(main())