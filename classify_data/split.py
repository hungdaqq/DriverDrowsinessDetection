import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(input_folder, output_folder, test_size=0.1):
    # Create output folders for training and testing sets
    # train_folder = os.path.join(output_folder, 'train')
    # test_folder = os.path.join(output_folder, 'test')
    # os.makedirs(train_folder, exist_ok=True)
    # os.makedirs(test_folder, exist_ok=True)

    # Get the list of files in the input folder
    file_list = os.listdir(input_folder)
    
    # Split the file list into training and testing sets
    _, test_files = train_test_split(file_list, test_size=test_size, random_state=42)

    # Copy training set files to the train folder
    # for file in train_files:
    #     src_path = os.path.join(input_folder, file)
    #     dst_path = os.path.join(train_folder, file)
    #     shutil.copyfile(src_path, dst_path)

    # Copy testing set files to the test folder
    for file in test_files:
        src_path = os.path.join(input_folder, file)
        dst_path = os.path.join(output_folder, file)
        shutil.move(src_path, dst_path)

# Specify the input folder containing the dataset
input_folder = './classify_data/train/Drowsy/'

# Specify the output folder where the train and test splits will be stored
output_folder = './classify_data/val/Drowsy/'

# Call the split_dataset function
split_dataset(input_folder, output_folder, test_size=0.1)
