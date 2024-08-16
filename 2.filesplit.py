import os
import shutil
from glob import glob
from sklearn.model_selection import train_test_split

# Set the path to the directory containing the images and label files
source_dir = "./data/w_g_01/"  # Replace with the actual path

# Define the directory names
train_dir = 'train'
valid_dir = 'valid'
test_dir = 'test'

# Define the function to create the directory
def create_dir(dir_name):
    # Check if a file exists with the directory name
    if os.path.isfile(dir_name):
        print(f"A file named '{dir_name}' exists. Removing the file.")
        os.remove(dir_name)
    # Create the directory if it doesn't exist
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
        print(f"Directory '{dir_name}' created.")

# Create the directories
create_dir(train_dir)
create_dir(valid_dir)
create_dir(test_dir)

# Get all JPG images in the folder
image_files = glob(os.path.join(source_dir, "*.jpg"))

# Split data into train, validation, and test sets
train_files, test_files = train_test_split(image_files, test_size=0.2, random_state=42)
val_files, test_files = train_test_split(test_files, test_size=0.5, random_state=42)

def move_files(files, destination_folder):
    for f in files:
        try:
            # Move the image file
            shutil.move(f, os.path.join(destination_folder, os.path.basename(f)))
            # Move the corresponding label file, if it exists
            label_file = f.replace('.jpg', '.txt')
            if os.path.isfile(label_file):
                shutil.move(label_file, os.path.join(destination_folder, os.path.basename(label_file)))
        except NotADirectoryError as e:
            print(f"NotADirectoryError: {e}")
        except FileNotFoundError as e:
            print(f"FileNotFoundError: {e}")

# Move the files
move_files(train_files, train_dir)
move_files(val_files, valid_dir)
move_files(test_files, test_dir)

print("Files have been moved to the respective 'train', 'valid', and 'test' directories.")
