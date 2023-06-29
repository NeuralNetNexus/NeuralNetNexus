import os
from pathlib import Path
import torch
from torchvision import models
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
import itertools
import matplotlib.pyplot as plt
import socketio
import requests
import sys
import zipfile
from models.cnn import CNN

project_id = os.getenv('PROJECT_ID')
n_splits = int(os.getenv('N_SPLITS'))
model = os.getenv('MODEL')

model_files = []

# Get Models from Bucket
for i in range(1, n_splits+1):
    response = requests.get(f"http://bucket-service/models/{project_id}_{i}.pth")
    if response.status_code == requests.codes.ok:
        model_name = f"/app/{project_id}_{i}.pth"
        with open(model_name, 'wb') as file:
            file.write(response.content)
            model_files.append(model_name)
    else:
        print('Error occurred while downloading the dataset. Status code:', response.status_code)
        sys.exit(5)

sio = socketio.Client()
sio.connect('ws://socket-service')
sio.emit('joinProject', project_id)

def get_model(model_name, num_classes):
    if model_name == 'VGG16':
        model = models.vgg16(pretrained=False) # create a vgg16 model architecture        
    elif model_name == 'ResNet18':
        model = models.resnet18(pretrained=False)
    elif model_name == 'EfficientNet V2S':
        model = models.efficientnet_v2_s(pretrained=False)
    elif model_name == 'SqueezeNet':
        model = models.squeezenet1_0(pretrained=False)
    elif model_name == 'CNN':
        model = CNN(num_classes=num_classes)
    return model

def train(model_files, model_name):

    numclasses = len(os.listdir(f"/app/{project_id}_test"))

    model = get_model(model_name, numclasses)

    params = model.state_dict()

    for k in params.keys():
        params[k] = torch.zeros_like(params[k])

    n_models = len(model_files)
    for model_path in model_files:
        model_state = torch.load(model_path) # load model parameters

        for k in params.keys():
            params[k] += model_state[k]

    # Average the parameters
    for k in params.keys():
        params[k] /= n_models

    # Load averaged parameters into the model architecture
    model_avg = get_model(model_name, numclasses)
    model_avg.load_state_dict(params)

    # Save the averaged model
    saved_model_path = f'/app/model_{project_id}.pth'
    torch.save(model_avg.state_dict(), saved_model_path)

    # Send Result to Bucket Service
    with open(saved_model_path, 'rb') as file:
        files = {'model': file}
        response = requests.post("http://bucket-service/models", files=files)


def compute_metrics(confusion_matrix):
    true_positives = np.diag(confusion_matrix)
    false_positives = np.sum(confusion_matrix, axis=0) - true_positives
    false_negatives = np.sum(confusion_matrix, axis=1) - true_positives

    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)
    f1_score = 2 * precision * recall / (precision + recall + 1e-10)

    return precision.mean(), recall.mean(), f1_score.mean()

def pil_loader(path):
    return Image.open(path).convert('RGB')

def test(test_dataset, model):
    dataset_test_obj = datasets.ImageFolder(root=test_dataset, transform=image_transforms["train"], loader=lambda path: pil_loader(path))
    test_loader = DataLoader(dataset_test_obj, batch_size=32, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    neuralnet = os.getenv('MODEL')
    size = (224, 224) if neuralnet is not "CNN" else (32, 32)

    image_transforms = [
        transforms.Resize(size),
        transforms.ToTensor(),
    ] 

    criterion = nn.CrossEntropyLoss()
    confusion_matrix = torch.zeros(num_classes, num_classes)
    idx_to_class = {v: k for k, v in dataset_test_obj.class_to_idx.items()}
    num_classes = len(idx_to_class)

    # Validation - No gradient tracking needed
    with torch.no_grad():

        # Set to evaluation mode
        model.eval()

        # Validation loop
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)

            # Compute the total loss for the batch and add it to valid_loss
            test_loss += loss.item() * inputs.size(0)

            # Calculate validation accuracy
            _, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            for t, p in zip(labels.view(-1), predictions.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            # Compute total accuracy in the whole batch and add to valid_acc
            test_acc += acc.item() * inputs.size(0)
    
    cm = confusion_matrix.cpu().numpy()
    plt.imshow(cm, cmap='gray_r')
    plt.colorbar()
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:.0f}".format(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > (cm.max() / 1.5) else "black")

    tick_marks = np.arange(len(confusion_matrix))
    plt.xticks(tick_marks, idx_to_class.values(), rotation=45)
    plt.yticks(tick_marks, idx_to_class.values())
    
    plt.tight_layout()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    conf_mat_path = f'/app/confusion_matrix_{project_id}.png'
    plt.savefig(conf_mat_path, bbox_inches="tight")

    # Send Result to Bucket Service
    with open(conf_mat_path, 'rb') as file:
        files = {'model': file}
        response = requests.post("http://bucket-service/models", files=files)
    
    plt.clf()
    
    avg_test_loss = test_loss/len(dataset_test_obj) 
    avg_test_acc = test_acc/len(dataset_test_obj)
    
    precision, recall, f1_score = compute_metrics(cm)
    
    indiv_acc = (confusion_matrix.diag()/confusion_matrix.sum(1)).numpy()
    
    print("Test: Loss - {:.4f}, Accuracy - {:.2f}%, Precision - {:.2f}%, Recall - {:.2f}%, F1-Score - {:.2f}%".format(avg_test_loss, avg_test_acc*100, precision*100, recall*100, f1_score*100))
    print("Test Per Class:")
    
    for key in idx_to_class:
        print(f"{idx_to_class[key]} - {round(indiv_acc[key] * 100, 2)}%")
        

    sio.emit('aggregatorMetrics', {
                "projectId": project_id,
                "loss": avg_test_loss,
                "accuracy": avg_test_acc*100+1, 
                "precision": precision*100,
                "recall": recall*100,
                "f1Score": f1_score*100,
            })

if __name__ == '__main__':

    try:    
        # Get dataset from bucket
        response = requests.get(f"http://bucket-service/datasets/{project_id}_test.zip")
        if response.status_code == requests.codes.ok:
            with open(f"/app/{project_id}_test.zip", 'wb') as file:
                file.write(response.content)
        else:
            print('Error occurred while downloading the dataset. Status code:', response.status_code)
            sys.exit(5)

        def unzip_folder(zip_path, extract_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)

        unzip_folder(f"/app/{project_id}_test.zip", f"/app/{project_id}_test")

        model = train(model_files, model)

        test_dataset = f"/app/{project_id}_test"

        test(test_dataset, model)
    except:
        sio.disconnect()
        exit(5)
    sio.disconnect()
    exit()