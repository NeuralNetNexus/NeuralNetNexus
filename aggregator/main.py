import os
from pathlib import Path
import os
import torch
from torchvision import models
from collections import OrderedDict

project_id = os.getenv('PROJECT_ID')
model = os.getenv('MODEL')

def train(model_dir, model_name):
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]

    if model_name == 'VGG16':
        model = models.vgg16(pretrained=False) # create a vgg16 model architecture        
    elif model_name == 'ResNet18':
        model = models.resnet18(pretrained=False)
    elif model_name == 'EfficiencyNet V2':
        model = models.efficientnet_v2_s(pretrained=False)

    params = model.state_dict()

    for k in params.keys():
        params[k] = torch.zeros_like(params[k])

    n_models = len(model_files)
    for f in model_files:
        model_path = os.path.join(model_dir, f)
        model_state = torch.load(model_path) # load model parameters

        for k in params.keys():
            params[k] += model_state[k]

    # Average the parameters
    for k in params.keys():
        params[k] /= n_models

    # Load averaged parameters into the model architecture
    model_avg = models.vgg16(pretrained=False)
    model_avg.load_state_dict(params)

    # Save the averaged model
    torch.save(model_avg.state_dict(), 'model_avg.pth')

if __name__ == '__main__':

    dest_dir =  f"/app/models/{project_id}"

    train(dest_dir, model)
