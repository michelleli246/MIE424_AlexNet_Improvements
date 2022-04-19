# imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

# datasets
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100

from alex import *

def evaluate(dl, device, model):
    total = 0
    correct = 0
    with torch.no_grad():
        for idx, (data, target) in enumerate(dl):
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            _, prediction = output.max(1)
            correct += (prediction == target).sum()
            total += prediction.size(0)
    return correct/total


def run_eval(model_path, activation):
    # transforms
    test_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((227,227)),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    # get datasets
    test_ds = CIFAR10("data/", train=False, download=True, transform=test_transform)

    #passing the train, val and test datasets to the dataloader
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AlexNetActivation(activation)
    model.cuda()

    model.load_state_dict(torch.load(model_path, map_location="cuda:0"))  # Choose whatever GPU device number you want

        
    test_result = evaluate(test_dl, device, model)
    print(test_result)

activation = "Tanh"
if __name__ == "__main__":

    if activation == "ReLU":
        model_save_path = "./models/relu_activation_model.pt"
        run_eval(model_save_path, nn.ReLU())

    if activation == "Sigmoid":
        model_save_path = "./models/sigmoid_activation_model.pt"
        run_eval(model_save_path, nn.Sigmoid())

    if activation == "LeakyReLU":
        model_save_path = "./models/leakyrelu_activation_model.pt"
        run_eval(model_save_path, nn.LeakyReLU())

    if activation == "Tanh":
        model_save_path = "./models/tanh_activation_model.pt"
        run_eval(model_save_path, nn.Tanh())

    if activation == "ELU":
        model_save_path = "./models/elu_activation_model.pt"
        run_eval(model_save_path, nn.ELU())