import os
import shutil
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


def is_file_modified(src, dst):
    """Check if the source file is newer than the destination file."""
    if not os.path.exists(dst):
        return True
    src_mod_time = os.path.getmtime(src)
    dst_mod_time = os.path.getmtime(dst)
    return src_mod_time > dst_mod_time


def copy_file(src, dst):
    """Copy the file if it's modified."""
    if is_file_modified(src, dst):
        shutil.copy2(src, dst)
        return f"Copied {src}"
    return f"Skipped {src}"


def walk_and_copy(src, dst, executor, futures):
    """Walk through the source and destination directories and copy files."""
    if os.path.isdir(src):
        if not os.path.exists(dst):
            os.makedirs(dst)
        for item in os.listdir(src):
            src_item = os.path.join(src, item)
            dst_item = os.path.join(dst, item)
            walk_and_copy(src_item, dst_item, executor, futures)
    else:
        future = executor.submit(copy_file, src, dst)
        futures.append(future)


# List of files/folders to back up
backup_files = [
    r"C:\Users\Blue\Desktop\Misc",
    r"C:\Users\Blue\Desktop\UIA.Resume",
    r"C:\Users\Blue\Desktop\Repos",
    r"C:\Users\Blue\Desktop\Servers",
    r"C:\Users\Blue\Desktop\DnD_Books",
]

# Backup directory
backup_dir = r"E:\Backup"

# Ensure the backup directory exists
os.makedirs(backup_dir, exist_ok=True)

print("Backup Started")

# Count the total number of files to copy
total_files = 0
for item in backup_files:
    if os.path.isdir(item):
        total_files += sum([len(files) for _, _, files in os.walk(item)])
    else:
        total_files += 1

# Perform parallel backup
with ThreadPoolExecutor() as executor:
    futures = []
    copied_files = 0  # Initialize the counter

    for item in backup_files:
        dst_path = os.path.join(backup_dir, os.path.basename(item))
        walk_and_copy(item, dst_path, executor, futures)

# Wait for all tasks to complete before listing copied files and total time taken
copied_files_list = []
start_time = datetime.datetime.now()

for future in as_completed(futures):
    result = future.result()
    if result.startswith("Copied"):
        copied_files_list.append(result)

end_time = datetime.datetime.now()
total_time_taken = end_time - start_time

print("\nList of Copied Files:")
for file in copied_files_list:
    print(file)

print("\nTotal Time Taken:", total_time_taken)

print("\nBackup completed to", backup_dir)

# Keep the window open (for running outside an IDE or without a .bat file)
input("Press Enter to exit...")
