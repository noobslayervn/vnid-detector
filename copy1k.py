import os
import shutil

source_dir = "1k_new_cccd"
destination_dir = "500_new_cccd"

# Ensure the destination directory exists, create if not
os.makedirs(destination_dir, exist_ok=True)

# Get a list of all files in the source directory
files = os.listdir(source_dir)

# Sort the files to ensure consistent ordering
files.sort()

# Iterate over the first 1000 files
for i, file in enumerate(files):
    if i >= 500:
        break
    source_file = os.path.join(source_dir, file)
    # Ensure it's a file and not a directory
    if os.path.isfile(source_file):
        # Construct the destination file path
        destination_file = os.path.join(destination_dir, file)
        # Copy the file to the destination directory
        shutil.move(source_file, destination_file)

print("First 500 image files moved successfully.")

