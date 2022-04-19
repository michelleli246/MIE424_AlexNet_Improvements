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

from models.alex import *

import matplotlib.pyplot as plt


def train_lr(train_dl, val_dl, epochs, lr, model_save_path):
    # check if cuda available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))

    # init model
    model = AlexNet()
    model = model.to(device)

    # For plots
    train_plot = []
    val_plot = []
    loss_plot = []

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # training loop
    for e in range(epochs):
        total_loss = 0

        # compute then backprop
        for idx, (data, target) in enumerate(train_dl):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # calculate accuracy
        train_acc = eval(train_dl, device, model)
        val_acc = eval(val_dl, device, model)

        train_plot.append(train_acc.item())
        val_plot.append(val_acc.item())
        loss_plot.append(total_loss)

        if e % 10 == 0:
            print('Epoch: {}  Loss: {}  Training Accuracy: {}  Validation Accuracy: {}'.format(
                e, total_loss, train_acc, val_acc))

    # save model
    torch.save(model.state_dict(), model_save_path)

    # plot and save
    plot(loss_plot, train_plot, val_plot, lr)


def main_lr(model_save_path, lr):
    # transforms
    train_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((227, 227)),
                                                      torchvision.transforms.RandomHorizontalFlip(
                                                          p=0.7),
                                                      torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    test_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((227, 227)),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    '''    
    # get datasets
    train_ds = CIFAR10("data/", train=True, download=True, transform=train_transform)
    val_size = 5000
    train_size = len(train_ds) - val_size
    train_ds, val_ds = random_split(train_ds, [train_size, val_size])
    test_ds = CIFAR10("data/", train=False, download=True, transform=test_transform)
    '''
    # get subset of dataset for training
    full_train_ds = CIFAR10("data/", train=True,
                            download=True, transform=train_transform)
    subset_train_size = len(full_train_ds)//5
    subset_train_ds, train_ds_pt2 = random_split(full_train_ds, [subset_train_size, len(
        full_train_ds) - subset_train_size], generator=torch.Generator().manual_seed(42))

    val_size = 1000
    train_size = subset_train_size - val_size

    train_ds, val_ds = random_split(subset_train_ds, [train_size, val_size])

    test_ds = CIFAR10("data/", train=False, download=True,
                      transform=test_transform)

    # passing the train, val and test datasets to the dataloader
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)

    # hyperparameters
    epochs = 25  # NB: epochs reduced from 50 to 25

    # train
    train_lr(train_dl, val_dl, epochs, lr, model_save_path)


def plot(loss_plot, train_plot, val_plot, label):
    plt.figure(1)
    plt.plot(loss_plot, label="lr=" + str(label))
    plt.legend()
    plt.figure(2)
    plt.plot(train_plot, label="lr=" + str(label))
    plt.legend()
    plt.figure(3)
    plt.plot(val_plot, label="lr=" + str(label))
    plt.legend()


def save_plot():
    plt.figure(1)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Rate Loss")
    plt.savefig("./images/lr_loss.png")

    plt.figure(2)
    plt.xlabel("Epoch")
    plt.ylabel("Train Accuracy")
    plt.title("Learning Rate Train Accuracy")
    plt.savefig("./images/lr_train.png")

    plt.figure(3)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Learning Rate Validation Accuracy")
    plt.savefig("./images/lr_val.png")


def main(lr):
    model_save_path = "./models/lr_+" + str(lr) + "_model.pt"
    main_lr(model_save_path, lr)


if __name__ == "__main__":

    for lr in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        main(lr)

    save_plot()
