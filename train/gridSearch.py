import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
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
from sklearn.model_selection import KFold


def pil_loader(path):
    return Image.open(path).convert('RGB')

def compute_metrics(confusion_matrix):
    true_positives = np.diag(confusion_matrix)
    false_positives = np.sum(confusion_matrix, axis=0) - true_positives
    false_negatives = np.sum(confusion_matrix, axis=1) - true_positives

    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)
    f1_score = 2 * precision * recall / (precision + recall + 1e-10)

    return precision.mean(), recall.mean(), f1_score.mean()

# Datasets
dataset_collection = [
    { "path": "./DatasetHalf", "channels": 3 },
]

image_transforms = {
    "train": transforms.Compose([
        transforms.Resize((228, 228)),
        transforms.RandomCrop(size=224),
        transforms.ToTensor(),
    ]),
    "valid": transforms.Compose([
        transforms.Resize((228, 228)),
        transforms.RandomCrop(size=224),
        transforms.ToTensor(),
    ]),
    "test": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
}

for i, dataset in enumerate(dataset_collection):
    dataset_path = dataset["path"]
    dataset_channels = dataset["channels"]

    dataset_train_path = os.path.join(dataset_path, 'train')
    dataset_test_path = os.path.join(dataset_path, 'test')
    dataset_valid_path = os.path.join(dataset_path, 'valid')

    # Normalized Dataset
    dataset_train_obj = datasets.ImageFolder(root=dataset_train_path, transform=image_transforms["train"], loader=lambda path: pil_loader(path))
    dataset_val_obj = datasets.ImageFolder(root=dataset_valid_path, transform=image_transforms["valid"], loader=lambda path: pil_loader(path))
    dataset_test_obj = datasets.ImageFolder(root=dataset_test_path, transform=image_transforms["test"], loader=lambda path: pil_loader(path))

    # kf = KFold(n_splits=5, shuffle=True)

    # Get a mapping of the indices to the class names, in order to see the output classes of the test images.
    idx_to_class = {v: k for k, v in dataset_train_obj.class_to_idx.items()}

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

        train_loader = DataLoader(dataset_train_obj, batch_size=hp["batch_size"], shuffle=True)
        val_loader = DataLoader(dataset_val_obj, batch_size=hp["batch_size"], shuffle=True)
        test_loader = DataLoader(dataset_test_obj, batch_size=hp["batch_size"], shuffle=True)

        train_data_size = len(dataset_train_obj)
        print(train_data_size)
        valid_data_size = len(dataset_val_obj)
        print(valid_data_size)
        test_data_size = len(dataset_test_obj)
    
        # Models
        model_collection = [
            models.SqueezeNet(num_classes=num_classes, num_channels=dataset_channels),
        ]

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

            print("==========  TEST  ==========")
            with open(os.path.join(history_path, 'test.log'), 'a+') as the_file:
                the_file.write('==========  TEST  ==========\n')

            # Validation - No gradient tracking needed
            with torch.no_grad():

                # Set to evaluation mode
                model_ft.eval()

                # Validation loop
                for inputs, labels in test_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Forward pass - compute outputs on input data using the model
                    outputs = model_ft(inputs)

                    # Compute loss
                    loss = criterion(outputs, labels)

                    # Compute the total loss for the batch and add it to valid_loss
                    test_loss += loss.item() * inputs.size(0)

                    # Calculate validation accuracy
                    ret, predictions = torch.max(outputs.data, 1)
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
            plt.savefig(os.path.join(history_path, 'confusion_matrix.png'), bbox_inches="tight")
            
            plt.clf()
            
            avg_test_loss = test_loss/test_data_size 
            avg_test_acc = test_acc/test_data_size
            
            precision, recall, f1_score = compute_metrics(cm)
            
            indiv_acc = (confusion_matrix.diag()/confusion_matrix.sum(1)).numpy()
            
            print("Test: Loss - {:.4f}, Accuracy - {:.2f}%, Precision - {:.2f}%, Recall - {:.2f}%, F1-Score - {:.2f}%".format(avg_test_loss, avg_test_acc*100, precision*100, recall*100, f1_score*100))
            print("Test Per Class:")
            
            for key in idx_to_class:
                print(f"{idx_to_class[key]} - {round(indiv_acc[key] * 100, 2)}%")
                
            print()

            with open(os.path.join(history_path, 'test.log'), 'a+') as the_file:
                the_file.write("Test: Loss - {:.4f}, Accuracy - {:.2f}%, Precision - {:.2f}%, Recall - {:.2f}%, F1-Score - {:.2f}%\n\n".format(avg_test_loss, avg_test_acc*100, precision*100, recall*100, f1_score*100))
                the_file.write('==========  TEST PER CLASS  ==========\n')
                
                for key in idx_to_class:
                    the_file.write(f"{idx_to_class[key]} - {round(indiv_acc[key] * 100, 2)}%\n")
