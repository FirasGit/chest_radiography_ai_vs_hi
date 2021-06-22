import os


def create_folder_structure(destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
