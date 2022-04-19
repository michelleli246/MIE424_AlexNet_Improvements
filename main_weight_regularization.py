# imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

import tensorflow as tf
import matplotlib as plt

import time

# datasets
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100

class AlexNet(nn.Module):
    # architecture
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels= 96, kernel_size= 11, stride=4, padding=0 )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride= 1, padding= 2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride= 1, padding= 1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.fc1  = nn.Linear(in_features= 9216, out_features= 4096)
        self.fc2  = nn.Linear(in_features= 4096, out_features= 4096)
        self.fc3 = nn.Linear(in_features=4096 , out_features=10)

    # set up network
    # add dropout here
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_w_regularization(train_dl, val_dl, epochs, lr, model_save_path, l1_weight = 0, l2_weight = 0):

    # check if cuda available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))

    # init model
    model = AlexNet()
    model = model.to(device)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_plot = []
    val_plot = []
    loss_plot = []

    # training loop
    for e in range(epochs):
        total_loss = 0
        total_crit_loss = 0
        start = time.time()

        # compute then backprop
        for idx, (data, target) in enumerate(train_dl):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)


            loss = criterion(output, target)
            crit_loss = loss.detach().clone()

            # Compute L1 L2 loss
            model_parameters = []
            for parameter in model.parameters():
                model_parameters.append(parameter.view(-1))
            model_parameters = torch.cat(model_parameters)
            
            l1 = 0
            l2 = 0
            if l1_weight != 0 and l2_weight == 0:
              l1 = torch.abs(model_parameters).sum()
            elif l2_weight != 0 and l1_weight == 0:
              l2 = torch.square(model_parameters).sum()
            else:
              l1 = torch.abs(model_parameters).sum()
              l2 = torch.square(model_parameters).sum()

            # Add L1 L2 to loss
            loss += l1*l1_weight
            loss += l2*l2_weight

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            total_crit_loss += crit_loss.item()

        
        # calculate accuracy
        train_acc = eval(train_dl, device, model)
        val_acc = eval(val_dl, device, model)

        train_plot.append(train_acc.item())
        val_plot.append(val_acc.item())
        loss_plot.append(total_loss)
        
        #if e % 5 == 0:
        print('Epoch: {}  CritLoss: {} Loss: {}  Training Accuracy: {}  Validation Accuracy: {}'.format(e, total_crit_loss, total_loss, train_acc, val_acc))

        end = time.time()
        print('time:',end - start)

    # save model
    torch.save(model.state_dict(), model_save_path+'_model.pt')

    #plot
    plot(loss_plot, train_plot, val_plot)

def plot(loss_plot, train_plot, val_plot):
    plt.figure(1)
    plt.plot(loss_plot)
    plt.legend()
    plt.figure(2)
    plt.plot(train_plot)
    plt.legend()
    plt.figure(3)
    plt.plot(val_plot)
    plt.legend()


def save_plot():
    plt.figure(1)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Weight Regularization Loss")
    plt.savefig("./images/weight_reg_loss.png")

    plt.figure(2)
    plt.xlabel("Epoch")
    plt.ylabel("Train Accuracy")
    plt.title("Weight Regularization Train Accuracy")
    plt.savefig("./images/weight_reg_train.png")

    plt.figure(3)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Weight Regularization Validation Accuracy")
    plt.savefig("./images/weight_reg_val.png")

def main_w_regularization(model_save_path, l1_weight, l2_weight):
    # transforms
    train_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((227, 227)),
                                                      torchvision.transforms.RandomHorizontalFlip(p=0.7),
                                                      torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    test_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((227,227)),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    # get datasets
    full_train_ds = CIFAR10("data/", train=True, download=True, transform=train_transform)
    subset_train_size = len(full_train_ds)//5
    subset_train_ds, train_ds_pt2 = random_split(full_train_ds, [subset_train_size,len(full_train_ds ) - subset_train_size], generator=torch.Generator().manual_seed(42))

    val_size = 1000
    train_size = subset_train_size - val_size

    train_ds, val_ds = random_split(subset_train_ds, [train_size, val_size])
    test_ds = CIFAR10("data/", train=False, download=True, transform=test_transform)

    #passing the train, val and test datasets to the dataloader
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)

    # hyperparameters
    epochs = 25
    lr = 1e-4

    # train
    train_w_regularization(train_dl, val_dl, epochs, lr, model_save_path, l1_weight, l2_weight)



# calculate accuracy
def eval(dl, device, model):
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

def run_eval(model_path):
    # transforms
    test_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((227,227)),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    # get datasets
    test_ds = CIFAR10("data/", train=False, download=True, transform=test_transform)

    #passing the train, val and test datasets to the dataloader
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AlexNet()

    # load model depending on if gpu available
    if tf.test.gpu_device_name() == '/device:GPU:0':
      model.load_state_dict(torch.load(model_path+'_model.pt', map_location="cuda:0"))  # Choose whatever GPU device number you want
    else:
      model.load_state_dict(torch.load(model_path+'_model.pt',map_location=torch.device('cpu')))
    model.to(device)
        
    test_result = eval(test_dl, device, model)
    print(test_result)



if __name__ == "__main__":
    experiment_hyperparameters = [[1e-4,0],[1e-6,0],[0,1e-3],[0,1e-4],[0,1e-5],[1e-4*0.05,1e-4*0.95],[1e-6*0.2,1e-4*0.8]]
    model_save_path = "./models/weight_regularization"
    for i in len(experiment_hyperparameters):
        main_w_regularization(model_save_path+'_exp'+str(i),experiment_hyperparameters[i][0],experiment_hyperparameters[i][1])
        run_eval(model_save_path+'_exp'+str(i))
    

