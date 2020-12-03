import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchsummary import summary
from torch.autograd import Variable
from torch.optim.sgd import SGD
import pickle
import pid
import os
import numpy as np
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from DNN_models import cifar10_CNN, cifar10_DenseNet, cifar10_ResNet18
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Hyper Parameters
input_size = 784
hidden_size = 1000
num_classes = 10
num_epochs = 5
batch_size = 100

'要进行对比实验的算法'
algorithm_labels = ['SGD', 'RMSprop', 'Adam', 'PID', 'Adam_self', 'RMSprop_self', 'Momentum', 'decade_PID', 'ID',
          'Adapid', 'Double_Adapid', 'Restrict_Adam', 'Logristrict_Adam', 'specPID', 'SVRG', 'SARAH']
'每种算法所对应的学习率'


learning_rates = [1, 0.001, 0.001, 0.2, 0.001, 0.001, 0.2, 1, 0.2, 0.001, 0.001, 0.001, 0.001, 0.1, 0.05, 0.05]

I = 3
I = float(I)
D = 30
D = float(D)

dataset_path = input('please input CIFAR10 path, nothing for default value \n')
if dataset_path == '':
    dataset_path = 'E:\\files\\programs\\Gitkraken\\cifar-10-batches-py\\'
for i in range(1,6):
    path = dataset_path + 'data_batch_' + str(i)
    with open(path, 'rb') as batch:
        dict = pickle.load(batch, encoding='bytes')
    if i == 1:
        images = dict[b'data']
        image_labels = dict[b'labels']
    else:
        images = np.concatenate([images, dict[b'data']], axis=0)
        image_labels = np.concatenate([image_labels, dict[b'labels']], axis=0)
path = dataset_path + 'test_batch'
with open(path, 'rb') as batch:
    dict = pickle.load(batch, encoding='bytes')
test_images = np.array(dict[b'data'])
test_image_labels = np.array(dict[b'labels'])
images = np.array(images)
image_labels = np.array(image_labels)
images = np.reshape(images, [-1, 3, 32, 32])
test_images = np.reshape(test_images, [-1, 3, 32, 32])
print(len(images))

class cifar10_dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.images = images
        self.labels = image_labels
        super(cifar10_dataset, self).__init__()
    def __getitem__(self, index):
        data = self.images[index]
        label = self.labels[index]
        return data, label
    def __len__(self):
        return len(self.images)

class cifar10_test_dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.images = test_images
        self.labels = test_image_labels
        super(cifar10_test_dataset, self).__init__()
    def __getitem__(self, index):
        data = self.images[index]
        label = self.labels[index]
        return data, label
    def __len__(self):
        return len(self.images)

train_loader = torch.utils.data.DataLoader(dataset=cifar10_dataset(), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=cifar10_test_dataset(), batch_size=batch_size, shuffle=True)
BGD_loader = torch.utils.data.DataLoader(dataset=cifar10_dataset(),batch_size=len(images),shuffle=True)


def training(model_sign=0, optimizer_sign=0, learning_rate=0.001, derivative=30):
    training_data = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
    if model_sign == 0:
        net = cifar10_DenseNet(num_classes)
        padding_sign = True
    elif model_sign == 1:
        net = cifar10_CNN(num_classes)
        padding_sign = False
    elif model_sign == 2:
        net = cifar10_ResNet18(num_classes)
        padding_sign = False
    else:
        raise ValueError('Not correct model sign')
    # net = Net(input_size, hidden_size, num_classes)
    net.cuda()
    net.train()
    # Loss and Optimizer
    oldnet_sign = False
    basicgrad_sign = False
    criterion = nn.CrossEntropyLoss()
    print('optimizer_sign:' + str(optimizer_sign))
    if optimizer_sign == 0:
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    elif optimizer_sign == 1:
        optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate)
    elif optimizer_sign == 2:
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    elif optimizer_sign == 3:
        optimizer = pid.PIDOptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9, I=I, D=derivative)
    elif optimizer_sign == 4:
        optimizer = pid.Adamoptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9)
    elif optimizer_sign == 5:
        optimizer = pid.RMSpropOptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001)
    elif optimizer_sign == 6:
        optimizer = pid.Momentumoptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9)
    elif optimizer_sign == 7:
        optimizer = pid.decade_PIDOptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9, I=I, D=derivative)
    elif optimizer_sign == 8:
        optimizer = pid.IDoptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9, I=I, D=derivative)
    elif optimizer_sign == 9:
        optimizer = pid.AdapidOptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9, I=I, D=derivative)
    elif optimizer_sign == 10:
        optimizer = pid.Double_Adaptive_PIDOptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9, I=I, D=derivative)
    elif optimizer_sign == 11:
        optimizer = pid.Restrict_Adamoptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9)
    elif optimizer_sign == 12:
        optimizer = pid.Logrestrict_Adamoptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9)
    elif optimizer_sign == 13:
        optimizer = pid.specPIDoptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9, I=I,D=derivative)
        oldnet_sign = True
    elif optimizer_sign == 14:
        optimizer = pid.SVRGoptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001)
        oldnet_sign = True
        basicgrad_sign = True
    elif optimizer_sign == 15:
        optimizer = pid.SARAHoptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001)
        oldnet_sign = True
        basicgrad_sign = True
    else:
        raise ValueError('Not correct algorithm symbol')
    if oldnet_sign == True:
        torch.save(net, 'net.pkl')
        old_net = torch.load('net.pkl')

    # Train the Model
    for epoch in range(num_epochs):

        train_loss_log = AverageMeter()
        train_acc_log = AverageMeter()
        val_loss_log = AverageMeter()
        val_acc_log = AverageMeter()
        for i, (images, labels) in enumerate(train_loader):
            if i % 100 == 0 and basicgrad_sign == True:
                for j, (all_images, all_labels) in enumerate(BGD_loader):
                    all_images = all_images.cuda().float()
                    if padding_sign == True:
                        all_images = all_images.view(-1, 3072)
                    all_labels = Variable(all_labels.cuda()).long()
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
                          % (epoch + 1, num_epochs, i + 1, len(images) // batch_size, train_loss_log.avg,
                             train_acc_log.avg))
            # Convert torch tensor to Variable
            images = images.cuda().float()
            if padding_sign == True:
                images = images.view(-1, 3072)
            labels = Variable(labels.cuda()).long()

            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer
            outputs = net(images)
            train_loss = criterion(outputs, labels)
            train_loss.backward()
            if oldnet_sign == True:
                old_outputs = old_net(images)
                old_loss = criterion(old_outputs, labels)
                old_loss.backward()
                old_params = list(old_net.parameters())
                old_grads = []
                for param in old_params:
                    old_grads.append(param.grad.detach())
                optimizer.get_oldgrad(old_grads)
            optimizer.step()
            if oldnet_sign == True and optimizer_sign != 8:
                torch.save(net, 'net.pkl')
                old_net = torch.load('net.pkl')
            prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
            train_loss_log.update(train_loss.data, images.size(0))
            train_acc_log.update(prec1, images.size(0))

            if (i + 1) % 30 == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Acc: %.8f'
                      % (epoch + 1, num_epochs, i + 1, len(images) // batch_size, train_loss_log.avg,
                         train_acc_log.avg))
                training_data['train_loss'].append(train_loss_log.avg.detach().cpu().numpy())
                training_data['train_acc'].append(train_acc_log.avg.detach().cpu().numpy())

        # Test the Model
        net.eval()
        correct = 0
        loss = 0
        total = 0
        for images, labels in test_loader:
            images = images.cuda().float()
            if padding_sign == True:
                images = images.view(-1, 3072)
            labels = Variable(labels).cuda().long()
            outputs = net(images)
            test_loss = criterion(outputs, labels)
            val_loss_log.update(test_loss.data, images.size(0))
            prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
            val_acc_log.update(prec1, images.size(0))

        #logger.append([learning_rate, train_loss_log.avg, val_loss_log.avg, train_acc_log.avg, val_acc_log.avg])
        print('Accuracy of the network on the 10000 test images: %.8f %%' % (val_acc_log.avg))
        print('Loss of the network on the 10000 test images: %.8f' % (val_loss_log.avg))
        training_data['val_loss'].append(val_loss_log.avg.detach().cpu().numpy())
        training_data['val_acc'].append(val_acc_log.avg.detach().cpu().numpy())
    #logger.close()
    #logger.plot()
    training_data['learning_rate'] = learning_rate
    return training_data

model_sign = int(input('please input model sign: \n 0 for Densenet, 1 for CNN, 2 for ResNet18 \nmodel_sign:'))

models = ['DenseNet', 'CNN', 'ResNet']


comparing_data = []
learning_rates = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.01]
derivatives = [-50, -10, -1, -0.1, 0.1, 1, 10, 50]

optimizer_sign = 9
derivative_sign = True

if derivative_sign == True:
    for i in range(len(derivatives)):
        for j in range(5):
            if j == 0:
                acc_data = np.array(training(model_sign=model_sign, optimizer_sign=optimizer_sign, derivative = derivatives[i])['train_acc'])
            else:
                acc_data += np.array(training(model_sign=model_sign, optimizer_sign=optimizer_sign, derivative = derivatives[i])['train_acc'])
        comparing_data.append(acc_data/5)
else:
    for i in range(len(learning_rates)):
        comparing_data.append(
            training(model_sign=model_sign, optimizer_sign=optimizer_sign, learning_rate=learning_rates[i]))

labels = ['SGD', 'RMSprop', 'Adam', 'PID', 'Adam_self', 'RMSprop_self', 'Momentum', 'decade_PID', 'ID',
          'Adapid', 'Double_Adapid', 'specPID', 'Restrict_Adam', 'SVRG', 'SARAH']





lrlabels = []
for i in range(len(derivatives)):
    if derivative_sign == False:
        lrlabels.append(labels[optimizer_sign] + ' lr=' + str(learning_rates[i]))
    else:
        lrlabels.append(labels[optimizer_sign] + ' d=' + str(derivatives[i]))



for i in range(len(comparing_data)):
    plt.plot(range(len(comparing_data[i])), comparing_data[i], label=labels[i])
plt.legend(lrlabels)
plt.title(models[model_sign] + ' CIFAR10 ')
plt.show()

a = 0
