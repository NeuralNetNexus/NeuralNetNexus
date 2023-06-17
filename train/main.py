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
    model = os.getenv('MODEL')
    JOB_COMPLETION_INDEX = int(os.getenv('JOB_COMPLETION_INDEX'))
    print("Starting training job with index:", JOB_COMPLETION_INDEX)

    train(file_path, dest_dir, model)
