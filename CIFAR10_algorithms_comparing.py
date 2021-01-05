import os


gpu_specify = input ('Please choose a device, 0~3 for single GPU, 4 for all GPUs, none for CPU: \n')
if gpu_specify == '':
    gpu_sign = 0
elif gpu_specify == '4':
    gpu_sign = 1
elif gpu_specify == '0' or gpu_specify == '1' or gpu_specify == '2' or gpu_specify == '3':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_specify
    gpu_sign = 2
else:
    raise ValueError('incorrect GPU symbol')

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchsummary import summary
from torch.autograd import Variable
from torch.optim.sgd import SGD
import pickle
import alpha_optimizers
import os
import numpy as np
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from DNN_models import cifar10_CNN, cifar10_DenseNet, cifar10_ResNet18
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

# Hyper Parameters
num_classes = 10
num_epochs = 500
batch_size = 1000

#dataset_sign = int(input('Please input a dataset sign, 0 for mnist, 1 for cifar10, 2 for imagenet'))
model_sign = int(input('please input model sign: \n 0 for Densenet, 1 for CNN, 2 for ResNet18 \nmodel_sign:'))
 #cifar10 dataset
dataset_path = 'cifar-10-batches-py/'
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
if model_sign == 2:
    images = images.transpose((0, 2, 3, 1))
    resized_images = []
    for i in range(len(images)):
        resized_images.append(np.array(Image.fromarray(images[i]).resize((224, 224), Image.BICUBIC)))
    images = np.array(resized_images).transpose((0, 3, 1, 2))
    test_images = test_images.transpose((0, 2, 3, 1))
    resized_images = []
    for i in range(len(test_images)):
        resized_images.append(np.array(Image.fromarray(test_images[i]).resize((224, 224), Image.BICUBIC)))
    test_images = np.array(resized_images).transpose((0, 3, 1, 2))
print('dataset extract completed,there are ' + str(len(images)) + ' images')

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

#testing functon
def training(model_sign=0, optimizer_sign=0, learning_rate=0.01, momentum=0.9, beta=0.999, alpha=1):
    training_data = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
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
    if gpu_sign == 1:
        net = torch.nn.DataParallel(net, device_ids=[0, 1, 2, 3])
    if gpu_sign != 0:
        net.cuda()
    net.train()
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    print('optimizer_sign:' + str(optimizer_sign))
    if optimizer_sign == 0:
        optimizer = alpha_optimizers.Adamoptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001,
                                                   momentum=momentum, beta=beta)
    elif optimizer_sign == 1:
        optimizer = alpha_optimizers.alpha_Adamoptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001,
                                                         momentum=momentum, beta=beta, alpha=alpha)
    elif optimizer_sign == 2:
        optimizer = alpha_optimizers.alpha_SGDoptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001,
                                                        momentum=momentum, alpha=alpha)
    elif optimizer_sign == 3:
        optimizer = alpha_optimizers.alpha_ascent_Adamoptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001,
                                                                momentum=momentum, beta=beta)
    elif optimizer_sign == 4:
        optimizer = alpha_optimizers.double_alpha_Adamoptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001,
                                                                momentum=momentum, beta=beta, alpha=alpha)
    elif optimizer_sign == 5:
        optimizer = alpha_optimizers.alpha2ascent_Adamoptimizer(net.parameters(), lr=learning_rate[0],sgd_lr=learning_rate[1], weight_decay=0.0001,
                                                                momentum=momentum, beta=beta)
    elif optimizer_sign == 6:
        optimizer = alpha_optimizers.alpha2_SGDoptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001,
                                                                momentum=momentum, alpha=alpha)
    elif optimizer_sign == 7:
        optimizer = alpha_optimizers.SGD_momentumoptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=momentum)
    elif optimizer_sign == 8:
        optimizer = alpha_optimizers.Adam_to_SGDoptimizer(net.parameters(), lr=learning_rate[0], sgd_lr=learning_rate[1], weight_decay=0.0001,
                                                          momentum=momentum, beta=beta)
    else:
        raise ValueError('Not correct algorithm symbol')

    # Train the Model
    for epoch in range(num_epochs):

        train_loss_log = AverageMeter()
        train_acc_log = AverageMeter()
        val_loss_log = AverageMeter()
        val_acc_log = AverageMeter()
        for i, (images, labels) in enumerate(train_loader):
            # Convert torch tensor to Variable
            if gpu_sign != 0:
                images = images.cuda()
                labels = labels.cuda()
            images = images.float()
            if padding_sign == True:
                images = images.view(-1, 3072)
            labels = Variable(labels).long()

            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer
            outputs = net(images)
            train_loss = criterion(outputs, labels)
            train_loss.backward()
            optimizer.step()
            prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
            train_loss_log.update(train_loss.data, images.size(0))
            train_acc_log.update(prec1, images.size(0))

            if (i + 1) % 20 == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Acc: %.8f'
                      % (epoch + 1, num_epochs, i + 1, 50000 / batch_size, train_loss_log.avg,
                         train_acc_log.avg))
                training_data['train_loss'].append(train_loss_log.avg.detach().cpu().numpy())
                training_data['train_acc'].append(train_acc_log.avg.detach().cpu().numpy())
        # Test the Model
        net.eval()
        correct = 0
        loss = 0
        total = 0
        for images, labels in test_loader:
            if gpu_sign != 0:
                images = images.cuda()
                labels = labels.cuda()
            images = images.float()
            if padding_sign:
                images = images.view(-1, 3072)
            labels = Variable(labels).long()
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


'Algorithms that can be choosed'
algorithm_labels = ['0.Adam', '1.alpha_adam', '2.alpha_SGD', '3.alpha_ascent_adam', '4.double_alpha_adam',
                    '5.alpha2ascent_adam', '6.alpha2_SGD', '7.SGD', '8.Adam_to_SGD']

task = int(input('please input a task, 0 for algorithm comparing, 1 for learning rate modify, '
                 '2 for alpha modify \n'))
if task == 0:
    test_algorithms = eval(input('please input testing algorithms, only list consist of int(algorithm sign) supported\n'))
    test_algorithms = [int(i) for i in test_algorithms]
    learning_rates = eval(input('please input learning rates, must corresponding to the algorithms \n'))
    if len(test_algorithms) < 1 or len(test_algorithms) != len(learning_rates):
        raise ValueError('lr and algorithms are not corresponding')
    alphas = eval(input('please input the list of testing alphas correspond to test algorithms \n'))
    if len(test_algorithms) != len(alphas):
        raise ValueError('alphas and algorithms are not corresponding')
    for i in range(len(test_algorithms)):
        if test_algorithms[i] == 4:
            if not isinstance(alphas[i], list):
                raise ValueError('algorithm4 need to input a list')
        else:
            alphas[i]  = float(alphas[i])
elif task == 1:
    test_algorithm = int(input('please input a single algorithm symbol \n'))
    learning_rates = eval(input('please input testing learning rates,only list supported \n'))
    if test_algorithm == 4:
        alphas = eval(input('please input a single alpha list for algorithm 4\n'))
    else:
        alphas = float(input('please input a single alpha \n'))
elif task == 2:
    test_algorithm = int(input('please input a single algorithm symbol \n'))
    learning_rates = eval(input('please input learning rates for the algorithm \n'))
    alphas = eval(input('please input alphas corresponding to learning rates, some algorithms may need to input lists \n'))
    if len(alphas) != len(learning_rates):
        raise ValueError('alphas and learning rates are not corresponding')
else:
    raise ValueError('not correct task symbol')
repeats = int(input('please input how many times to repeat \n'))

show_symbol = eval(input('please choose what to show, 0 for accuracy, 1 for loss, 2 for training_err,'
                         ' 3 for mean derivatives, support multiple chioce. please input an list \n'))
for i in show_symbol:
    i = int(i)
    if not(i == 0 or i == 1 or i == 2 or i == 3 or i == 4):
        raise  ValueError('incorrect show symbol')

shows = ['acc', 'loss', 'training_err']
models = ['DenseNet', 'CNN', 'ResNet']
comparing_datas = [[] for i in show_symbol]
comparing_data = [[] for i in show_symbol]
test_algorithm_labels = [[] for i in show_symbol]
if task == 0:
    for i in range(len(test_algorithms)):
        for j in range(repeats):
            output = training(model_sign=model_sign, optimizer_sign=test_algorithms[i],
                              learning_rate=learning_rates[i],alpha=alphas[i])
            for a in range(len(show_symbol)):
                if j == 0:
                    if show_symbol[a] == 0:
                        comparing_data[a] = np.array(output['train_acc'])
                    elif show_symbol[a] == 1:
                        comparing_data[a] = np.array(output['train_loss'])
                    elif show_symbol[a] == 2:
                        comparing_data[a] = 100 - np.array(output['train_acc'])
                    elif show_symbol[a] == 3:
                        comparing_data[a] = np.array(output['val_acc'])
                    else:
                        comparing_data[a] = np.array(output['val_loss'])
                else:
                    if show_symbol[a] == 0:
                        comparing_data[a] += np.array(output['train_acc'])
                    elif show_symbol[a] == 1:
                        comparing_data[a] += np.array(output['train_loss'])
                    elif show_symbol[a] == 2:
                        comparing_data[a] += 100 - np.array(output['train_acc'])
                    elif show_symbol[a] == 3:
                        comparing_data[a] += np.array(output['val_acc'])
                    else:
                        comparing_data[a] += np.array(output['val_loss'])
        for a in range(len(show_symbol)):
            comparing_datas[a].append(np.array(comparing_data[a]) / repeats)
            test_algorithm_labels[a].append(
                algorithm_labels[test_algorithms[i]] + ' learning_rate=' + str(learning_rates[i]) + ' alpha=' + str(alphas[i]))
elif task == 1:
    for i in range(len(learning_rates)):
        for j in range(repeats):
            output = training(model_sign=model_sign, optimizer_sign=test_algorithm,
                              learning_rate=learning_rates[i], alpha=alphas)
            for a in range(len(show_symbol)):
                if j == 0:
                    if show_symbol[a] == 0:
                        comparing_data[a] = np.array(output['train_acc'])
                    elif show_symbol[a] == 1:
                        comparing_data[a] = np.array(output['train_loss'])
                    elif show_symbol[a] == 2:
                        comparing_data[a] = 100 - np.array(output['train_acc'])
                    elif show_symbol[a] == 3:
                        comparing_data[a] = np.array(output['val_acc'])
                    else:
                        comparing_data[a] = np.array(output['val_loss'])
                else:
                    if show_symbol[a] == 0:
                        comparing_data[a] += np.array(output['train_acc'])
                    elif show_symbol[a] == 1:
                        comparing_data[a] += np.array(output['train_loss'])
                    elif show_symbol[a] == 2:
                        comparing_data[a] += 100 - np.array(output['train_acc'])
                    elif show_symbol[a] == 3:
                        comparing_data[a] += np.array(output['val_acc'])
                    else:
                        comparing_data[a] += np.array(output['val_loss'])
        for a in range(len(show_symbol)):
            comparing_datas[a].append(np.array(comparing_data[a]) / repeats)
            test_algorithm_labels[a].append(
                algorithm_labels[test_algorithm] + ' learning_rate=' + str(learning_rates[i]) + 'alpha=' + str(alphas))
elif task == 2:
    for i in range(len(learning_rates)):
        for j in range(repeats):
            output = training(model_sign=model_sign, optimizer_sign=test_algorithm,
                              learning_rate=learning_rates[i],
                              alpha=alphas[i])
            for a in range(len(show_symbol)):
                if j == 0:
                    if show_symbol[a] == 0:
                        comparing_data[a] = np.array(output['train_acc'])
                    elif show_symbol[a] == 1:
                        comparing_data[a] = np.array(output['train_loss'])
                    elif show_symbol[a] == 2:
                        comparing_data[a] = 100 - np.array(output['train_acc'])
                    elif show_symbol[a] == 3:
                        comparing_data[a] = np.array(output['val_acc'])
                    else:
                        comparing_data[a] = np.array(output['cal_loss'])
                else:
                    if show_symbol[a] == 0:
                        comparing_data[a] += np.array(output['train_acc'])
                    elif show_symbol[a] == 1:
                        comparing_data[a] += np.array(output['train_loss'])
                    elif show_symbol[a] == 2:
                        comparing_data[a] += 100 - np.array(output['train_acc'])
                    elif show_symbol[a] == 3:
                        comparing_data[a] += np.array(output['val_acc'])
                    else:
                        comparing_data[a] += np.array(output['val_loss'])
        for a in range(len(show_symbol)):
            comparing_datas[a].append(np.array(comparing_data[a]) / repeats)
            test_algorithm_labels[a].append(
                algorithm_labels[test_algorithm] + ' learning_rate=' + str(learning_rates[i]) + ' alpha=' + str(alphas[i]))
save_sign = 10
for a in range(len(show_symbol)):
    for i in range(len(comparing_datas[a])):
        plt.plot(range(len(comparing_datas[a][i])), comparing_datas[a][i])
    plt.legend(test_algorithm_labels[a])

    plt.title(models[model_sign] + ' CIFAR10, ' + shows[show_symbol[a]])
    plt.savefig('data/matplotlib/' + str(save_sign))
    save_sign += 1
    plt.show()
    plt.cla()

a = 0




