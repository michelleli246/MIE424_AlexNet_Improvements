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


class AlexNet(nn.Module):
    # architecture
    def __init__(self, dropout):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride= 1, padding= 2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride= 1, padding= 1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=9216, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=10)

        self.dropout = nn.Dropout(dropout)

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
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def train(train_dl, val_dl, epochs, lr, model_save_path, dropout):
    # check if cuda available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))

    # init model
    model = AlexNet(dropout)
    model = model.to(device)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # collect
    train_accuracy = []
    val_accuracy = []
    losses = []

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

        # collect
        train_accuracy.append(train_acc)
        val_accuracy.append(val_acc)
        losses.append(total_loss)

        if e % 1 == 0:
            print('Epoch: {}  Loss: {}  Training Accuracy: {}  Validation Accuracy: {}'.format(e + 1, total_loss, train_acc, val_acc))

    print(train_accuracy)
    print(val_accuracy)
    print(losses)

    # save model
    # torch.save(model.state_dict(), model_save_path)


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
    return correct / total


def main(model_save_path):
    torch.manual_seed(424)
    # transforms
    train_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((227, 227)),
                                                      torchvision.transforms.RandomHorizontalFlip(p=0.7),
                                                      torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    test_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((227,227)),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # get datasets
    train_ds = CIFAR10("data/", train=True, download=True, transform=train_transform)
    val_size = 5000
    train_size = len(train_ds) - val_size
    train_ds, val_ds = random_split(train_ds, [train_size, val_size])
    test_ds = CIFAR10("data/", train=False, download=True, transform=test_transform)

    # create dataloaders
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)

    # hyperparameters
    epochs = 50
    lr = 1e-4
    dropouts = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    for dropout in dropouts:
        # train
        print('Dropout = {}'.format(dropout))
        train(train_dl, val_dl, epochs, lr, model_save_path, dropout)


main('dropout_50.pt')


# Baseline
# [tensor(0.6038, device='cuda:0'), tensor(0.7168, device='cuda:0'), tensor(0.7961, device='cuda:0'), tensor(0.8290, device='cuda:0'), tensor(0.8565, device='cuda:0'), tensor(0.9069, device='cuda:0'), tensor(0.9134, device='cuda:0'), tensor(0.9320, device='cuda:0'), tensor(0.9565, device='cuda:0'), tensor(0.9606, device='cuda:0'), tensor(0.9728, device='cuda:0'), tensor(0.9671, device='cuda:0'), tensor(0.9741, device='cuda:0'), tensor(0.9778, device='cuda:0'), tensor(0.9814, device='cuda:0'), tensor(0.9778, device='cuda:0'), tensor(0.9770, device='cuda:0'), tensor(0.9854, device='cuda:0'), tensor(0.9889, device='cuda:0'), tensor(0.9864, device='cuda:0'), tensor(0.9891, device='cuda:0'), tensor(0.9874, device='cuda:0'), tensor(0.9891, device='cuda:0'), tensor(0.9897, device='cuda:0'), tensor(0.9918, device='cuda:0'), tensor(0.9911, device='cuda:0'), tensor(0.9928, device='cuda:0'), tensor(0.9894, device='cuda:0'), tensor(0.9922, device='cuda:0'), tensor(0.9822, device='cuda:0'), tensor(0.9916, device='cuda:0'), tensor(0.9868, device='cuda:0'), tensor(0.9932, device='cuda:0'), tensor(0.9912, device='cuda:0'), tensor(0.9893, device='cuda:0'), tensor(0.9899, device='cuda:0'), tensor(0.9894, device='cuda:0'), tensor(0.9942, device='cuda:0'), tensor(0.9948, device='cuda:0'), tensor(0.9958, device='cuda:0'), tensor(0.9918, device='cuda:0'), tensor(0.9944, device='cuda:0'), tensor(0.9952, device='cuda:0'), tensor(0.9927, device='cuda:0'), tensor(0.9944, device='cuda:0'), tensor(0.9960, device='cuda:0'), tensor(0.9927, device='cuda:0'), tensor(0.9953, device='cuda:0'), tensor(0.9920, device='cuda:0'), tensor(0.9955, device='cuda:0')]
# [tensor(0.5792, device='cuda:0'), tensor(0.6916, device='cuda:0'), tensor(0.7580, device='cuda:0'), tensor(0.7772, device='cuda:0'), tensor(0.7900, device='cuda:0'), tensor(0.8138, device='cuda:0'), tensor(0.8060, device='cuda:0'), tensor(0.8146, device='cuda:0'), tensor(0.8228, device='cuda:0'), tensor(0.8170, device='cuda:0'), tensor(0.8304, device='cuda:0'), tensor(0.8174, device='cuda:0'), tensor(0.8186, device='cuda:0'), tensor(0.8250, device='cuda:0'), tensor(0.8254, device='cuda:0'), tensor(0.8148, device='cuda:0'), tensor(0.8194, device='cuda:0'), tensor(0.8210, device='cuda:0'), tensor(0.8306, device='cuda:0'), tensor(0.8234, device='cuda:0'), tensor(0.8324, device='cuda:0'), tensor(0.8184, device='cuda:0'), tensor(0.8282, device='cuda:0'), tensor(0.8258, device='cuda:0'), tensor(0.8264, device='cuda:0'), tensor(0.8310, device='cuda:0'), tensor(0.8278, device='cuda:0'), tensor(0.8266, device='cuda:0'), tensor(0.8280, device='cuda:0'), tensor(0.8126, device='cuda:0'), tensor(0.8278, device='cuda:0'), tensor(0.8186, device='cuda:0'), tensor(0.8286, device='cuda:0'), tensor(0.8228, device='cuda:0'), tensor(0.8270, device='cuda:0'), tensor(0.8288, device='cuda:0'), tensor(0.8242, device='cuda:0'), tensor(0.8210, device='cuda:0'), tensor(0.8314, device='cuda:0'), tensor(0.8328, device='cuda:0'), tensor(0.8290, device='cuda:0'), tensor(0.8274, device='cuda:0'), tensor(0.8338, device='cuda:0'), tensor(0.8248, device='cuda:0'), tensor(0.8328, device='cuda:0'), tensor(0.8310, device='cuda:0'), tensor(0.8278, device='cuda:0'), tensor(0.8360, device='cuda:0'), tensor(0.8266, device='cuda:0'), tensor(0.8268, device='cuda:0')]
# [1038.7346581816673, 687.0366413593292, 518.3485818207264, 417.63892552256584, 338.74156887829304, 271.3499235510826, 220.80304336547852, 171.82024786248803, 136.69298854283988, 112.8364943517372, 91.15106889046729, 75.01521743647754, 65.85234308661893, 61.44774362957105, 55.65738846035674, 48.19729728868697, 44.30900684295921, 40.54391537792981, 37.638827190428856, 34.86896065599285, 37.888750729500316, 30.273171848471975, 31.013872361014364, 30.37313775277289, 28.466041992680402, 26.35840476641897, 25.92469632985012, 25.097259608621243, 21.068767578762674, 25.759129835409112, 21.972187765320996, 22.531471765978495, 20.159393176523736, 19.69771213678905, 17.866394263897746, 23.525862390677503, 18.917103201234568, 19.560515123761434, 16.49915473823784, 16.49314793065031, 18.862766233032744, 15.889059195925029, 16.58035736070451, 19.25352836782986, 13.365400937548316, 17.936053916255332, 15.668834551561304, 15.301158475926059, 13.836110712552909, 14.445883066251554]


# Dropout 0.5
# [tensor(0.5625, device='cuda:0'), tensor(0.6668, device='cuda:0'), tensor(0.7464, device='cuda:0'), tensor(0.7820, device='cuda:0'), tensor(0.8126, device='cuda:0'), tensor(0.8348, device='cuda:0'), tensor(0.8560, device='cuda:0'), tensor(0.8794, device='cuda:0'), tensor(0.8959, device='cuda:0'), tensor(0.8915, device='cuda:0'), tensor(0.9164, device='cuda:0'), tensor(0.9171, device='cuda:0'), tensor(0.9289, device='cuda:0'), tensor(0.9345, device='cuda:0'), tensor(0.9347, device='cuda:0'), tensor(0.9504, device='cuda:0'), tensor(0.9541, device='cuda:0'), tensor(0.9372, device='cuda:0'), tensor(0.9551, device='cuda:0'), tensor(0.9565, device='cuda:0'), tensor(0.9635, device='cuda:0'), tensor(0.9675, device='cuda:0'), tensor(0.9676, device='cuda:0'), tensor(0.9713, device='cuda:0'), tensor(0.9677, device='cuda:0')]
# [tensor(0.5628, device='cuda:0'), tensor(0.6606, device='cuda:0'), tensor(0.7280, device='cuda:0'), tensor(0.7494, device='cuda:0'), tensor(0.7754, device='cuda:0'), tensor(0.7838, device='cuda:0'), tensor(0.7992, device='cuda:0'), tensor(0.8148, device='cuda:0'), tensor(0.8226, device='cuda:0'), tensor(0.8192, device='cuda:0'), tensor(0.8228, device='cuda:0'), tensor(0.8260, device='cuda:0'), tensor(0.8304, device='cuda:0'), tensor(0.8210, device='cuda:0'), tensor(0.8306, device='cuda:0'), tensor(0.8380, device='cuda:0'), tensor(0.8390, device='cuda:0'), tensor(0.8118, device='cuda:0'), tensor(0.8386, device='cuda:0'), tensor(0.8342, device='cuda:0'), tensor(0.8344, device='cuda:0'), tensor(0.8420, device='cuda:0'), tensor(0.8348, device='cuda:0'), tensor(0.8428, device='cuda:0'), tensor(0.8306, device='cuda:0')]
# [1096.3744159340858, 750.0955083966255, 590.8865588307381, 494.7969789505005, 427.34784249961376, 375.2950272113085, 340.0386323481798, 296.4979215711355, 265.3825814202428, 240.27627430856228, 214.9415097013116, 190.38129100762308, 172.98123240470886, 160.0636817459017, 139.62578059732914, 135.74323216453195, 116.84405370056629, 109.79362386465073, 105.68001048639417, 95.18298363755457, 90.2170212417841, 90.09148185886443, 79.86191488988698, 79.81129879876971, 74.73436638806015]


# Dropout 0.25
# [tensor(0.5942, device='cuda:0'), tensor(0.7157, device='cuda:0'), tensor(0.7818, device='cuda:0'), tensor(0.8189, device='cuda:0'), tensor(0.8320, device='cuda:0'), tensor(0.8560, device='cuda:0'), tensor(0.8888, device='cuda:0'), tensor(0.9184, device='cuda:0'), tensor(0.9216, device='cuda:0'), tensor(0.9366, device='cuda:0'), tensor(0.9544, device='cuda:0'), tensor(0.9652, device='cuda:0'), tensor(0.9634, device='cuda:0'), tensor(0.9649, device='cuda:0'), tensor(0.9694, device='cuda:0'), tensor(0.9679, device='cuda:0'), tensor(0.9762, device='cuda:0'), tensor(0.9687, device='cuda:0'), tensor(0.9764, device='cuda:0'), tensor(0.9698, device='cuda:0'), tensor(0.9820, device='cuda:0'), tensor(0.9823, device='cuda:0'), tensor(0.9845, device='cuda:0'), tensor(0.9799, device='cuda:0'), tensor(0.9834, device='cuda:0')]
# [tensor(0.5938, device='cuda:0'), tensor(0.7028, device='cuda:0'), tensor(0.7512, device='cuda:0'), tensor(0.7846, device='cuda:0'), tensor(0.7908, device='cuda:0'), tensor(0.7916, device='cuda:0'), tensor(0.8086, device='cuda:0'), tensor(0.8320, device='cuda:0'), tensor(0.8236, device='cuda:0'), tensor(0.8342, device='cuda:0'), tensor(0.8380, device='cuda:0'), tensor(0.8424, device='cuda:0'), tensor(0.8338, device='cuda:0'), tensor(0.8366, device='cuda:0'), tensor(0.8332, device='cuda:0'), tensor(0.8352, device='cuda:0'), tensor(0.8406, device='cuda:0'), tensor(0.8370, device='cuda:0'), tensor(0.8310, device='cuda:0'), tensor(0.8328, device='cuda:0'), tensor(0.8404, device='cuda:0'), tensor(0.8340, device='cuda:0'), tensor(0.8366, device='cuda:0'), tensor(0.8384, device='cuda:0'), tensor(0.8416, device='cuda:0')]
# [1060.503801882267, 699.4709518551826, 543.1430513560772, 444.1778651177883, 376.33100475370884, 321.31298683583736, 272.45697143673897, 223.58054164797068, 188.26630306243896, 161.25406707078218, 140.6039291601628, 114.6320465359604, 104.05981612671167, 95.29899668321013, 82.0307302037254, 73.72390232700855, 69.59531416418031, 64.74120204456267, 56.771057724021375, 55.947477631038055, 51.19383996515535, 48.82694814610295, 44.76889376388863, 45.43905262288172, 42.62375429086387]


# Dropout 0.75
# [tensor(0.5052, device='cuda:0'), tensor(0.5943, device='cuda:0'), tensor(0.7008, device='cuda:0'), tensor(0.7366, device='cuda:0'), tensor(0.7707, device='cuda:0'), tensor(0.7981, device='cuda:0'), tensor(0.8122, device='cuda:0'), tensor(0.8367, device='cuda:0'), tensor(0.8552, device='cuda:0'), tensor(0.8556, device='cuda:0'), tensor(0.8815, device='cuda:0'), tensor(0.8873, device='cuda:0'), tensor(0.8911, device='cuda:0'), tensor(0.9077, device='cuda:0'), tensor(0.9141, device='cuda:0'), tensor(0.9218, device='cuda:0'), tensor(0.9312, device='cuda:0'), tensor(0.9242, device='cuda:0'), tensor(0.9367, device='cuda:0'), tensor(0.9375, device='cuda:0'), tensor(0.9416, device='cuda:0'), tensor(0.9518, device='cuda:0'), tensor(0.9554, device='cuda:0'), tensor(0.9529, device='cuda:0'), tensor(0.9590, device='cuda:0')]
# [tensor(0.5018, device='cuda:0'), tensor(0.5820, device='cuda:0'), tensor(0.6842, device='cuda:0'), tensor(0.7114, device='cuda:0'), tensor(0.7342, device='cuda:0'), tensor(0.7510, device='cuda:0'), tensor(0.7584, device='cuda:0'), tensor(0.7856, device='cuda:0'), tensor(0.7898, device='cuda:0'), tensor(0.7954, device='cuda:0'), tensor(0.8074, device='cuda:0'), tensor(0.8140, device='cuda:0'), tensor(0.8008, device='cuda:0'), tensor(0.8152, device='cuda:0'), tensor(0.8198, device='cuda:0'), tensor(0.8140, device='cuda:0'), tensor(0.8236, device='cuda:0'), tensor(0.8110, device='cuda:0'), tensor(0.8154, device='cuda:0'), tensor(0.8176, device='cuda:0'), tensor(0.8232, device='cuda:0'), tensor(0.8178, device='cuda:0'), tensor(0.8298, device='cuda:0'), tensor(0.8210, device='cuda:0'), tensor(0.8316, device='cuda:0')]
# [1211.9211132526398, 853.4573330879211, 685.2029541134834, 581.7963583171368, 512.1119505316019, 458.5680074095726, 414.893427670002, 375.9533249735832, 341.6584891676903, 313.5803529471159, 286.0767394453287, 258.710555376485, 239.59459333121777, 215.79963582754135, 203.01798232644796, 187.7762483805418, 174.0774585455656, 163.11060521006584, 151.5994314327836, 137.960801448673, 131.4743135869503, 123.51748422347009, 118.46833960525692, 106.36871267668903, 110.16178838163614]
