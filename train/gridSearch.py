import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import itertools
import time
import os
import copy
from PIL import Image
import models
import random
from torch.utils.data import random_split
import shutil
import socketio
from kubernetes import client, config


config.load_kube_config()

# Create an instance of the Kubernetes API client
api_client = client.CoreV1Api()

# Retrieve the pod name from the environment variable
pod_name = os.getenv("HOSTNAME")

# Retrieve the namespace from the pod's service account token
namespace_path = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
with open(namespace_path, "r") as file:
    namespace = file.read()

project_id = os.getenv('PROJECT_ID')
job_completion_index = os.getenv('JOB_COMPLETION_INDEX')


sio = socketio.Client()
sio.connect('ws://socket-service')
sio.emit('joinProject', project_id)


def pil_loader(path):
    return Image.open(path).convert('RGB')

dataset_name = f"/app/datasets/{project_id}_{job_completion_index+1}"

# Datasets
dataset_collection = [
    { "path": dataset_name, "channels": 3 },
]

image_transforms = [
    transforms.Resize((228, 228)),
    transforms.RandomCrop(size=224),
    transforms.ToTensor(),
]

for i, dataset in enumerate(dataset_collection):
    
    dataset_path = dataset["path"]
    dataset_channels = dataset["channels"]
    
    dataset_obj = datasets.ImageFolder(root=dataset_path, transform=transforms.Compose(image_transforms), loader=lambda path: pil_loader(path))

    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'val')

    # Create the train and validation folders if they don't exist
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

    # Get the class-to-index mapping from the dataset object
    class_to_idx = dataset_obj.class_to_idx

    # Create class folders in the train and validation folders
    for class_name in class_to_idx.keys():
        class_train_path = os.path.join(train_path, class_name)
        class_val_path = os.path.join(val_path, class_name)
        os.makedirs(class_train_path, exist_ok=True)
        os.makedirs(class_val_path, exist_ok=True)

    # Move the images to the train and validation class folders
    for image_path, class_index in dataset_obj.imgs:
        class_name = dataset_obj.classes[class_index]
        if random.random() < 0.8:  # 80% for training, adjust the split ratio as needed
            destination_path = os.path.join(train_path, class_name, os.path.basename(image_path))
        else:
            destination_path = os.path.join(val_path, class_name, os.path.basename(image_path))
        shutil.move(image_path, destination_path)

    # Create the train and validation datasets
    train_dataset = datasets.ImageFolder(root=train_path, transform=transforms.Compose(image_transforms), loader=lambda path: pil_loader(path))
    val_dataset = datasets.ImageFolder(root=val_path, transform=transforms.Compose(image_transforms), loader=lambda path: pil_loader(path))

    # Get a mapping of the indices to the class names, in order to see the output classes of the test images.
    idx_to_class = {v: k for k, v in dataset_obj.class_to_idx.items()}

    num_classes = len(idx_to_class)

    # Hyperparameters
    hps = {
        "lr": [0.001],
        "batch_size": [32],
        "epochs": [1],
    }
    hps_iter = [dict(zip(hps.keys(), values)) for values in itertools.product(*hps.values())]

    # Grid search
    for j, hp in enumerate(hps_iter):

        train_loader = DataLoader(train_dataset, batch_size=hp["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=hp["batch_size"], shuffle=True)

        train_data_size = len(train_dataset)
        print(train_data_size)
        valid_data_size = len(val_dataset)
        print(valid_data_size)
    
        # Models
        neuralnet = os.getenv('MODELS')
        if neuralnet == "SqueezeNet":
            model_collection = [models.SqueezeNet(num_classes=num_classes, num_channels=dataset_channels)]
        elif neuralnet == "ResNet18":
            models.ResNet(resnet_version=18, num_classes=num_classes, num_channels=dataset_channels),
        elif neuralnet == "VGG16":
            model_collection = [models.VGG(num_classes=num_classes, num_channels=dataset_channels)]
        elif neuralnet == "EfficientNet V2S":
            model_collection = [models.EfficientNet(num_classes=num_classes, num_channels=dataset_channels),]    
                

        model_collection = [models.SqueezeNet(num_classes=num_classes, num_channels=dataset_channels)]

        for k, model in enumerate(model_collection):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Send the model to the processing unit
            model_ft = model.to(device)

            dataset_name = os.path.split(dataset_path)[1]
            history_path = f"./output/{dataset_name}-{model_ft.__class__.__name__.lower()}-epochs-{hp['epochs']}-lr-{hp['lr']}-batch-{hp['batch_size']}-optim-adam-cross"
            if not os.path.exists(history_path):
                os.makedirs(history_path)
                
            params_to_update = model_ft.model.parameters()
            if k != 2:
                if dataset_channels == 3:
                    params_to_update = []
                    for name, param in model_ft.model.named_parameters():
                        if param.requires_grad == True:
                            params_to_update.append(param)
            
            # Observe that all parameters are being optimized
            optimizer_ft = optim.Adam(params_to_update, lr=hp["lr"], betas=(0.9, 0.999))
            #optimizer_ft = optim.SGD(params_to_update, lr=hp["lr"], momentum=0.9)

            # Setup the loss fxn
            criterion = nn.CrossEntropyLoss()
            
            # TRAIN NETWORK
            start = time.time()
            history = []
            best_loss = 100000.0
            best_epoch = None
            best_model = None

            print(f"==========  TRAIN  ==========")
            with open(os.path.join(history_path, 'train.log'), 'a+') as the_file:
                the_file.write(f'==========  TRAIN  ==========\n')

            for epoch_i in range(hp["epochs"]):
                epoch_start = time.time()
                print("Epoch: {}/{}".format(epoch_i+1, hp["epochs"]))
                
                # Set to training mode
                model_ft.train()
                
                # Loss and Accuracy within the epoch
                train_loss = 0.0
                train_acc = 0.0
                
                valid_loss = 0.0
                valid_acc = 0.0
                
                for (inputs, labels) in train_loader:

                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    # Clean existing gradients
                    optimizer_ft.zero_grad()
                    
                    # Forward pass - compute outputs on input data using the model
                    outputs = model_ft(inputs)
                    
                    # Compute loss
                    loss = criterion(outputs, labels)
                    
                    # Backpropagate the gradients
                    loss.backward()
                    
                    # Update the parameters
                    optimizer_ft.step()
                    
                    # Compute the total loss for the batch and add it to train_loss
                    train_loss += loss.item() * inputs.size(0)
                    
                    # Compute the accuracy
                    ret, predictions = torch.max(outputs.data, 1)
                    correct_counts = predictions.eq(labels.data.view_as(predictions))
                    
                    # Convert correct_counts to float and then compute the mean
                    acc = torch.mean(correct_counts.type(torch.FloatTensor))
                    
                    # Compute total accuracy in the whole batch and add to train_acc
                    train_acc += acc.item() * inputs.size(0)

                
                # Validation - No gradient tracking needed
                with torch.no_grad():

                    # Set to evaluation mode
                    model_ft.eval()

                    # Validation loop
                    for inputs, labels in val_loader:
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        # Forward pass - compute outputs on input data using the model
                        outputs = model_ft(inputs)

                        # Compute loss
                        loss = criterion(outputs, labels)

                        # Compute the total loss for the batch and add it to valid_loss
                        valid_loss += loss.item() * inputs.size(0)

                        # Calculate validation accuracy
                        ret, predictions = torch.max(outputs.data, 1)
                        correct_counts = predictions.eq(labels.data.view_as(predictions))

                        # Convert correct_counts to float and then compute the mean
                        acc = torch.mean(correct_counts.type(torch.FloatTensor))

                        # Compute total accuracy in the whole batch and add to valid_acc
                        valid_acc += acc.item() * inputs.size(0)

                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_epoch = epoch_i
                    best_model = copy.deepcopy(model_ft.state_dict())

                # Find average training loss and training accuracy
                avg_train_loss = train_loss/train_data_size
                avg_train_acc = train_acc/train_data_size

                # Find average training loss and training accuracy
                avg_valid_loss = valid_loss/valid_data_size 
                avg_valid_acc = valid_acc/valid_data_size

                history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
                        
                epoch_end = time.time()
            
                log = "\tTraining: Loss - {:.4f}, Accuracy - {:.2f}%\n\tValidation: Loss - {:.4f}, Accuracy - {:.2f}%\n\tTime: {:.4f}s".format(avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start)
                print(log)
                with open(os.path.join(history_path, 'train.log'), 'a+') as the_file:
                    the_file.write(f'Epoch {epoch_i+1}:\n{log}\n')

                # Save if the model has best accuracy till now
                if epoch_i == hp["epochs"] - 1:
                    model_ft.load_state_dict(best_model)
                    # torch.save(model_ft, os.path.join(history_path, f'model_fold_{fold}.pt'))
                    torch.save(model_ft, os.path.join(history_path, f'model.pt'))
                    log = f"\nBest Model from Epoch {best_epoch+1}"
                    print(log)
                    with open(os.path.join(history_path, 'train.log'), 'a+') as the_file:
                        the_file.write(log + "\n\n")

                # Retrieve pod metrics
                metrics = api_client.read_namespaced_pod_metrics(pod_name, namespace)

                # Find the CPU usage for the first container in the pod
                container_metrics = metrics.containers[0]
                cpu_usage = container_metrics.usage["cpu"]
                ram_usage = container_metrics.usage["memory"]

            sio.emit('trainingMetrics', { "train_index": job_completion_index, 
                                            "epoch": epoch_i+1,
                                            "train_accuracy": avg_train_acc*100,
                                                "train_loss": avg_train_loss,
                                                "val_accuracy": avg_valid_acc*100,
                                                "val_loss": avg_valid_loss,
                                                "cpu_usage": cpu_usage,
                                                "ram_usage": ram_usage
                                                }, project_id)

            history = np.array(history)

            # Save val accuracy to file
            np.savetxt(os.path.join(history_path, 'accuracy_values.txt'), history[:,1], fmt='%.4f', delimiter=';')

            plt.plot(history[:,0:2])
            plt.legend(['Tr Loss', 'Val Loss'])
            plt.xlabel('Epoch Number')
            plt.ylabel('Loss')
            plt.ylim(0,2)
            plt.savefig(os.path.join(history_path, 'loss_curve.png'))

            plt.clf()

            plt.plot(history[:,2:4])
            plt.legend(['Tr Accuracy', 'Val Accuracy'])
            plt.xlabel('Epoch Number')
            plt.ylabel('Accuracy')
            plt.ylim(0,1)
            plt.savefig(os.path.join(history_path, 'accuracy_curve.png'))

            plt.clf()

            test_acc = 0.0
            test_loss = 0.0

            confusion_matrix = torch.zeros(num_classes, num_classes)