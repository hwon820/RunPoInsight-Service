import os
from glob import glob

# Set your directory path containing the images and label files
source_dir = "./data/w_g_01/"  # Replace with the actual path to your folder

# Get all JPG images in the folder
image_files = glob(os.path.join(source_dir, "*.jpg"))

# Define a function to create a new name given the old name and a counter
# This function should implement the renaming logic you want
def create_new_name(old_name, counter):
    # Example renaming logic: add a prefix 'image_' and a counter to the name
    new_base_name = f"w_g_01_{counter}"
    return new_base_name

# Initialize a counter for the new file names
counter = 1

# Start the renaming process
for image_file in image_files:
    # Create the new name
    new_base_name = create_new_name(image_file, counter)
    
    # Get the full path for the new .jpg file
    new_image_name = os.path.join(source_dir, new_base_name + '.jpg')
    
    # Rename the .jpg file
    os.rename(image_file, new_image_name)
    
    # Check if a corresponding .txt file exists
    txt_file = image_file.replace('.jpg', '.txt')
    if os.path.isfile(txt_file):
        # Get the full path for the new .txt file
        new_txt_name = os.path.join(source_dir, new_base_name + '.txt')
        
        # Rename the .txt file
        os.rename(txt_file, new_txt_name)
    
    # Increment the counter for the next file name
    counter += 1

print("Files have been renamed successfully.")
