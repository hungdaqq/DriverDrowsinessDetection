import os
import shutil

# Source and destination directories
source_dir = './Output/23/Alert/'
destination_dir = './classify_data/traincnn/Alert/'

# Iterate over files in the source directory
for filename in os.listdir(source_dir):
    source_path = os.path.join(source_dir, filename)
    
    # Skip directories
    if not os.path.isfile(source_path):
        continue
    
    # Generate a new filename
    new_filename = '23_' + filename
    
    # Destination path for the file
    destination_path = os.path.join(destination_dir, new_filename)
    
    # Move the file to the destination and rename it
    shutil.copy(source_path, destination_path)
    print(f"Copied '{filename}' to '{new_filename}' in the destination directory.")