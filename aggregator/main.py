import os
from pathlib import Path
import random

def train(file_path, dest_dir, model):
    # Ensure the destination directory exists
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    try:
        number = random.randint(0, 10)
        print("Generated number:", number)
        if number > 5:
            raise ValueError("Fatal error: Number is greater than 5.")
    except ValueError as e:
        print(e)

if __name__ == '__main__':
    file_path = "/app/datasets/dataset.zip"
    dest_dir =  "/app/datasets"
    try:
        model = os.getenv('MODEL')
        print("Starting aggregator job")
    except Exception as e:
        print(e)
        model = "default"

    train(file_path, dest_dir, model)
