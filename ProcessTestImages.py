import helpers.edge_detect.bg_rem as bgrem
import shutil
import os
import pathlib

def remove_directory(directory_path):
    """
    Removes a directory and its contents, working across platforms.

    Args:
        directory_path: The path to the directory to remove.
    """

    path = pathlib.Path(directory_path)

    if path.exists():
        try:
            shutil.rmtree(path)
            print(f"Directory '{directory_path}' removed successfully.")
        except OSError as e:
            print(f"Error removing directory '{directory_path}': {e}")
    else:
        print(f"Directory '{directory_path}' does not exist.")

remove_directory("Trial-AI-Base-Images\\bg_removed")
for file in os.listdir("Trial-AI-Base-Images"):
    if file.endswith(".jpg") or file.endswith(".png"):
        image_path = os.path.join("Trial-AI-Base-Images", file)
        bgrem.process_image(image_path, mode="base")
        print(f"Background removed from {file}")