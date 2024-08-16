import os
from glob import glob

# Set the path to the directory containing the files
source_dir = "./test"  # Replace with the actual path to your folder

# Create new directories for .jpg and .txt files within the source directory
jpg_dir = os.path.join(source_dir, "images")
txt_dir = os.path.join(source_dir, "labels")

# Create the directories if they don't exist
os.makedirs(jpg_dir, exist_ok=True)
os.makedirs(txt_dir, exist_ok=True)

# Get all JPG images and TXT files in the folder
image_files = glob(os.path.join(source_dir, "*.jpg"))
text_files = glob(os.path.join(source_dir, "*.txt"))

# Move the .jpg files
for image_file in image_files:
    # Define the destination path for the image file
    dest_path = os.path.join(jpg_dir, os.path.basename(image_file))
    # Move the file
    os.rename(image_file, dest_path)

# Move the .txt files
for text_file in text_files:
    # Define the destination path for the text file
    dest_path = os.path.join(txt_dir, os.path.basename(text_file))
    # Move the file
    os.rename(text_file, dest_path)

print("Files have been organized into separate folders.")
