import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from optimizers.pid import SARAHoptimizer
from utils import Logger, AverageMeter, accuracy
import torch.nn.functional as F
# Hyper Parameters 
input_size = 784
hidden_size = 1000
num_classes = 10
num_epochs = 20
batch_size = 100
learning_rate = 0.01

I=3
I = float(I)
D = 100
D = float(D)


logger = Logger('pid.txt', title='mnist')
logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

# MNIST Dataset 
train_dataset = dsets.MNIST(root='./data', 
                            train=True, 
                            transform=transforms.ToTensor(),  
                            download=True)

test_dataset = dsets.MNIST(root='./data', 
                           train=False, 
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

BGD_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=len(train_dataset),
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# Neural Network Model (1 hidden layer)
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = torch.nn.Conv2d(128, 256, 3, 1, 1)
        self.dense = torch.nn.Linear(256, 256)
        self.maxpool = torch.nn.MaxPool2d(2, 2, 0)
        self.averagepool=torch.nn.AvgPool2d(3)
        self.outlayer = torch.nn.Linear(256, num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.averagepool(x)
        x = x.view((x.size(0), -1))
        x = self.dense(x)
        x = self.relu(x)
        out = self.outlayer(x)

        return out


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        return out


net = Net(input_size, hidden_size, num_classes)
#net = Net(input_size, hidden_size, num_classes)
net.cuda()   
net.train()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()  
optimizer = SARAHoptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9)
torch.save(net, 'net.pkl')
old_net = torch.load('net.pkl')
# Train the Model
for epoch in range(num_epochs):

    train_loss_log = AverageMeter()
    train_acc_log = AverageMeter()
    val_loss_log = AverageMeter()
    val_acc_log = AverageMeter()
    for i, (images, labels) in enumerate(train_loader):
        if i % 100 == 0:
            for j, (all_images, all_labels) in enumerate(BGD_loader):
                all_images = all_images.view(-1, 28 * 28).cuda()
                all_labels = Variable(all_labels.cuda())
                optimizer.zero_grad()  # zero the gradient buffer
                outputs = net(all_images)
                train_loss = criterion(outputs, all_labels)
                train_loss.backward()
                params = list(net.parameters())
                grads = []
                for param in params:
                    grads.append(param.grad.detach())
                optimizer.get_basicgrad(grads)
                optimizer.step()
                prec1, prec5 = accuracy(outputs.data, all_labels.data, topk=(1, 5))
                train_loss_log.update(train_loss.data, all_images.size(0))
                train_acc_log.update(prec1, all_images.size(0))
                torch.save(net, 'net.pkl')
                old_net = torch.load('net.pkl')
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Acc: %.8f'
                      % (epoch + 1, num_epochs, i+1, len(train_dataset) // batch_size, train_loss_log.avg,
                         train_acc_log.avg))
        # Convert torch tensor to Variable
        images = images.view(-1, 28 * 28).cuda()
        labels = Variable(labels.cuda())
        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(images)
        train_loss = criterion(outputs, labels)
        train_loss.backward()
        old_outputs = old_net(images)
        old_loss = criterion(old_outputs, labels)
        old_loss.backward()
        old_params = list(old_net.parameters())
        old_grads = []
        for param in old_params:
            old_grads.append(param.grad.detach())
        optimizer.get_oldgrad(old_grads)
        optimizer.step()
        torch.save(net, 'net.pkl')
        old_net = torch.load('net.pkl')
        prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
        train_loss_log.update(train_loss.data, images.size(0))
        train_acc_log.update(prec1, images.size(0))
        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Acc: %.8f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, train_loss_log.avg,
                     train_acc_log.avg))

    # Test the Model
    net.eval()
    correct = 0
    loss = 0
    total = 0
    for images, labels in test_loader:
        images = images.view(-1, 28 * 28).cuda()
        labels = Variable(labels).cuda()
        outputs = net(images)
        test_loss = criterion(outputs, labels)
        val_loss_log.update(test_loss.data, images.size(0))
        prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
        val_acc_log.update(prec1, images.size(0))

    logger.append([learning_rate, train_loss_log.avg, val_loss_log.avg, train_acc_log.avg, val_acc_log.avg])
    print('Accuracy of the network on the 10000 test images: %.8f %%' % (val_acc_log.avg))
    print('Loss of the network on the 10000 test images: %.8f' % (val_loss_log.avg))

logger.close()
logger.plot()

