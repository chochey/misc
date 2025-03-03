import os
import sys
import time
import threading
import queue
import subprocess
import logging
import argparse
import signal
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ---------------------- Configuration ---------------------- #
class Config:
    MAIN_DIR = r"G:\Downloads\complete"
    SCRIPT_PATH = r"/Python Scripts/media_organizer/searchMovieDBandRenamesFolders.py"
    CHECK_INTERVAL = 1  # Seconds between size checks
    STABLE_DURATION = 30  # Seconds path must remain stable
    DEBOUNCE_DELAY = 5  # Seconds to debounce events
    REQUIRED_MODULES = ["rapidfuzz"]  # Add other required modules if needed

# ---------------------- Logging Setup ---------------------- #
def setup_logging(log_level):
    """Set up logging to console only."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.StreamHandler()  # Only log to console
        ]
    )

# ---------------------- Dependency Check ---------------------- #
def check_and_install_dependencies(python_executable):
    """Check and install required modules."""
    import importlib.util
    for module in Config.REQUIRED_MODULES:
        if importlib.util.find_spec(module) is None:
            logging.warning(f"Module '{module}' not found. Attempting to install...")
            try:
                subprocess.run([python_executable, "-m", "pip", "install", module], check=True)
                logging.info(f"Successfully installed '{module}'")
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to install '{module}': {e}. Please run 'pip install {module}' manually.")
                return False
    return True

# ---------------------- Stability Check ---------------------- #
def is_path_stable(path, stable_duration=Config.STABLE_DURATION, check_interval=Config.CHECK_INTERVAL, retries=3):
    """Check if a file or directory (and its contents) is stable using modification times."""
    def is_file_stable(file_path):
        for attempt in range(retries):
            try:
                last_mtime = os.stat(file_path).st_mtime
                stable_time = 0
                while stable_time < stable_duration:
                    time.sleep(check_interval)
                    current_mtime = os.stat(file_path).st_mtime
                    if current_mtime == last_mtime:
                        stable_time += check_interval
                    else:
                        stable_time = 0
                        last_mtime = current_mtime
                return True
            except Exception as e:
                logging.error(f"Attempt {attempt + 1}/{retries} failed for {file_path}: {e}")
                if attempt < retries - 1:
                    time.sleep(check_interval)
                else:
                    return False

    path = os.path.realpath(path)
    if os.path.isfile(path):
        return is_file_stable(path)
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for f in files:
                file_path = os.path.join(root, f)
                if not is_file_stable(file_path):
                    return False
        return True
    return False

# ---------------------- Event Handler ---------------------- #
class FileMoveHandler(FileSystemEventHandler):
    def __init__(self, event_queue, event_types, exclude_paths, debounce_delay=Config.DEBOUNCE_DELAY):
        self.event_queue = event_queue
        self.last_event_time = {}
        self.pending_events = {}
        self.debounce_delay = debounce_delay
        self.event_types = set(event_types)
        self.exclude_paths = set(os.path.realpath(p) for p in exclude_paths)

    def on_created(self, event):
        if "created" in self.event_types:
            self.queue_event(event.src_path)

    def on_modified(self, event):
        if "modified" in self.event_types:
            self.queue_event(event.src_path)

    def queue_event(self, path):
        path = os.path.realpath(path)
        if any(path.startswith(excl) for excl in self.exclude_paths):
            logging.debug(f"Excluded path: {path}")
            return
        current_time = time.time()
        parent_dir = os.path.dirname(path)
        if parent_dir in self.last_event_time and (current_time - self.last_event_time[parent_dir]) < self.debounce_delay:
            self.pending_events.setdefault(parent_dir, set()).add(path)
            logging.debug(f"Debouncing event for {path}")
            return
        self.last_event_time[parent_dir] = current_time
        if parent_dir in self.pending_events:
            self.pending_events[parent_dir].add(path)
            self.event_queue.put((parent_dir, self.pending_events.pop(parent_dir)))
        else:
            self.event_queue.put((parent_dir, {path}))
        logging.info(f"Queued event for directory: {parent_dir}")

# ---------------------- Processor ---------------------- #
def process_queue(event_queue, script_path, python_executable, stop_event):
    """Process queued events and run the script on stable directories."""
    processed_dirs = set()
    while not stop_event.is_set():
        try:
            parent_dir, paths = event_queue.get(timeout=1)
            logging.debug(f"Starting processing for {parent_dir} with paths: {paths}")
            if parent_dir in processed_dirs:
                logging.debug(f"Skipping already processed directory: {parent_dir}")
                continue
            if is_path_stable(parent_dir):
                logging.info(f"Directory '{parent_dir}' is stable. Running script...")
                if not os.path.exists(script_path):
                    logging.error(f"Script path '{script_path}' does not exist. Please verify the path.")
                    break
                retries = 3
                for attempt in range(retries):
                    try:
                        subprocess.run([python_executable, script_path, "--main-dir", parent_dir],
                                     check=True, timeout=300)
                        logging.info(f"Script completed successfully for {parent_dir}")
                        processed_dirs.add(parent_dir)
                        break
                    except FileNotFoundError as e:
                        logging.error(f"Cannot find Python executable or script at '{python_executable}' or '{script_path}': {e}")
                        break
                    except subprocess.TimeoutExpired:
                        logging.error(f"Script timed out for {parent_dir} on attempt {attempt + 1}/{retries}")
                    except subprocess.CalledProcessError as e:
                        if "ModuleNotFoundError" in str(e.output):
                            logging.error(f"Dependency error for {parent_dir}: {e.output}. Please ensure all modules (e.g., rapidfuzz) are installed with 'pip install rapidfuzz'")
                            break
                        logging.error(f"Script failed for {parent_dir} on attempt {attempt + 1}/{retries}: {e}")
                    if attempt < retries - 1:
                        time.sleep(5)
                else:
                    logging.error(f"All retries failed for {parent_dir}")
            else:
                logging.warning(f"Directory '{parent_dir}' did not become stable.")
            logging.debug(f"Finished processing for {parent_dir}")
        except queue.Empty:
            continue
        except FileNotFoundError as e:
            logging.error(f"File not found error in queue processing: {e}. Check Python executable or script path.")
        except Exception as e:
            logging.error(f"Unexpected error in queue processing: {e}")

# ---------------------- Main ---------------------- #
def main(args):
    setup_logging(args.log_level)
    python_executable = sys.executable  # Use current Python interpreter
    if not check_and_install_dependencies(python_executable):
        logging.error("Required dependencies missing. Exiting.")
        return

    event_queue = queue.Queue()
    stop_event = threading.Event()
    event_handler = FileMoveHandler(event_queue, args.event_types, args.exclude_paths, args.debounce_delay)
    observer = Observer()
    observer.schedule(event_handler, args.main_dir, recursive=True)
    observer.start()
    logging.info(f"Monitoring directory: {args.main_dir}")

    processor_thread = threading.Thread(target=process_queue, args=(event_queue, args.script_path, python_executable, stop_event))
    processor_thread.daemon = True
    processor_thread.start()

    def handle_shutdown(signum, frame):
        logging.info(f"Received signal {signum}. Stopping observer...")
        stop_event.set()
        observer.stop()

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    try:
        while True:
            time.sleep(1)
    except Exception as e:
        logging.error(f"Unexpected error in main loop: {e}")
    finally:
        stop_event.set()
        observer.stop()
        processor_thread.join(timeout=10)
        if processor_thread.is_alive():
            logging.warning("Processor thread did not terminate cleanly")
        observer.join()
        logging.info("Monitoring stopped")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Media folder watcher")
    parser.add_argument("--main-dir", default=Config.MAIN_DIR, help="Directory to monitor")
    parser.add_argument("--script-path", default=Config.SCRIPT_PATH, help="Path to media organizer script")
    parser.add_argument("--stable-duration", type=int, default=Config.STABLE_DURATION, help="Seconds for stability")
    parser.add_argument("--check-interval", type=int, default=Config.CHECK_INTERVAL, help="Seconds between checks")
    parser.add_argument("--debounce-delay", type=int, default=Config.DEBOUNCE_DELAY, help="Seconds to debounce events")
    parser.add_argument("--event-types", nargs="+", default=["created", "modified"],
                        choices=["created", "modified"], help="Event types to monitor")
    parser.add_argument("--exclude-paths", nargs="+", default=[], help="Paths to exclude")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    args = parser.parse_args()
    main(args)