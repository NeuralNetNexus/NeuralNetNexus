import os
import zipfile
from pathlib import Path

def split_zip(file_path, dest_dir, parts):
    # Ensure the destination directory exists
    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    # Open the zip file
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        # Get the list of files in the zip
        file_list = zip_ref.namelist()

        # Calculate the number of files per part
        files_per_part = len(file_list) // parts

        # Split the files into parts
        for i in range(parts):
            part_file_list = file_list[i*files_per_part:(i+1)*files_per_part]
            # Create a new zip file for each part
            with zipfile.ZipFile(f"{dest_dir}/part{i+1}.zip", 'w') as part_zip:
                # Add the files from the part file list to the new zip
                for file in part_file_list:
                    part_zip.writestr(file, zip_ref.read(file))

if __name__ == '__main__':
    file_path = "/app/datasets/dataset.zip"
    dest_dir =  "/app/datasets"
    parts = int(os.getenv('PARTS'))

    split_zip(file_path, dest_dir, parts)
