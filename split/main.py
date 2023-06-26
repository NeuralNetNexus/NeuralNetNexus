import os
import zipfile
import shutil
import glob
import requests
import socketio
import sys

sio = socketio.Client()

project_id = os.getenv('PROJECT_ID')
    
sio.connect('ws://socket-service')
sio.emit('joinProject', project_id)

def unzip_folder(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

def zip_folder(folder_path, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, folder_path))
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                zipf.write(dir_path, os.path.relpath(dir_path, folder_path))

def count_images(folder_path):
    image_files = glob.glob(os.path.join(folder_path, '**/*.jpg'), recursive=True)
    image_files += glob.glob(os.path.join(folder_path, '**/*.jpeg'), recursive=True)
    image_files += glob.glob(os.path.join(folder_path, '**/*.png'), recursive=True)
    image_files += glob.glob(os.path.join(folder_path, '**/*.gif'), recursive=True)
    image_files += glob.glob(os.path.join(folder_path, '**/*.bmp'), recursive=True)
    
    return len(image_files)

def split_zip(pvc_path, project_id):
    # CONSTANTS
    file_path = f"{pvc_path}/{project_id}.zip"

    # Unzip the folder
    dataset_path = os.path.join(pvc_path, project_id)
    unzip_folder(file_path, dataset_path)

    # Train and Test Paths
    train_path = os.path.join(dataset_path, "training")
    train_path = train_path if os.path.exists(train_path) else os.path.join(dataset_path, "train")
    test_path = os.path.join(dataset_path, "testing")
    test_path = test_path if os.path.exists(test_path) else os.path.join(dataset_path, "test")

    #shutil.copytree(test_path, os.path.join(pvc_path, f"{project_id}_test"))
    zip_folder(test_path, os.path.join(pvc_path, f"{project_id}_test.zip"))

    # Calculate the ratio and split size based on total images
    total_images = count_images(train_path)
    ratio = max(1, total_images // 10000)

    try:
        data = {
            'splits': ratio
        }
        requests.patch(f"http://backend-service/projects/{project_id}/n-splits", json=data)
        sio.emit('projectStatus', { "n_batch": ratio})

    except:
        print("Error sending the number of splits to the backend-service or websocket")


    # Get the list of classes
    class_list = os.listdir(train_path)

    # Iterate over each class
    for idx, class_name in enumerate(class_list):
        class_path = os.path.join(train_path, class_name)

        print(f"Progress: {idx / len(class_list) * 100}%")
        
        # Get the list of images in the class folder
        images_list = os.listdir(class_path)
        total_images = len(images_list)

        split_size = max(1, total_images // ratio)
        
        # Create a new dataset folder for each split
        for split_num in range(1, ratio + 1):

            # Create a new directory
            split_name = f"{project_id}_{split_num}/{class_name}"
            split_dir = os.path.join(pvc_path, split_name)
            os.makedirs(split_dir, exist_ok=True)
            
            # Determine the range of images for the current split
            start_index = (split_num - 1) * split_size
            end_index = split_num * split_size
            
            # Move the corresponding images to the split dataset folder
            for i in range(start_index, end_index):
                if i < total_images:
                    image_name = images_list[i]
                    image_path = os.path.join(class_path, image_name)
                    shutil.copy(image_path, split_dir)

    print(f"Progress: 100%")
    print(f"Zipping results...")
    for split_num in range(1, ratio + 1):
        path = os.path.join(pvc_path, f"{project_id}_{split_num}")
        zip_path = "{path}.zip"
        zip_folder(path, zip_path)

        # Send Result to Bucket Service
        with open(zip_path, 'rb') as file:
            files = {'dataset': file}
            response = requests.post("http://bucket-service/datasets", files=files)
        
        # Check the response status
        if response.status_code == requests.codes.ok:
            print(f'File {zip_path} uploaded successfully.')
        else:
            print(f'Error occurred while uploading the file {zip_path}. Status code:', response.status_code)
            sys.exit(5)

        shutil.rmtree(path)

    shutil.rmtree(dataset_path)


if __name__ == '__main__':
    pvc_path = "/app"

    # Get file from bucket
    response = requests.get(f"http://bucket-service/datasets/{project_id}.zip")

    if response.status_code == requests.codes.ok:
        with open(f"{pvc_path}/{project_id}.zip", 'wb') as file:
            file.write(response.content)
    else:
        print('Error occurred while downloading the dataset. Status code:', response.status_code)
        sys.exit(5)

    split_zip(pvc_path, project_id)
    sio.disconnect()
    print(f"Done!")
    sys.exit()
