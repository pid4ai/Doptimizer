import torch
import torchvision
from torch.autograd import Variable
import torch.utils.data.dataloader as Data
import matplotlib.pyplot as plt
from torchsummary import summary
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class conv_block(torch.nn.Module):
    def __init__(self, input_channels, output_channels):
        super(conv_block, self).__init__()
        if output_channels > input_channels:
            self.stride = 2
            self.sample = torch.nn.Conv2d(input_channels, output_channels, 1, 2, 0)
        else:
            self.stride = 1
        self.conv1 = torch.nn.Conv2d(input_channels, output_channels, 3, self.stride, 1)
        self.conv2 = torch.nn.Conv2d(output_channels, output_channels, 3, 1, 1)
        self.bn = torch.nn.BatchNorm2d(output_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x1 = x
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        if self.stride == 2:
            x1 = self.sample(x1)
        x1 = self.bn(x1)
        x = x + x1
        x = self.relu(x)
        return x

class bottle_block(torch.nn.Module):
    def __init__(self, input_channels, down_channels, output_channels, stride = 1):
        super(bottle_block, self).__init__()
        self.stride = stride
        self.conv1 = torch.nn.Conv2d(input_channels, down_channels, 1, 1, 0)
        self.conv2 = torch.nn.Conv2d(down_channels, down_channels, 3, self.stride, 1)
        self.conv3 = torch.nn.Conv2d(down_channels, output_channels, 1, 1, 0)
        self.sample = torch.nn.Conv2d(input_channels, output_channels, 1, self.stride, 0)
        self.bn1 = torch.nn.BatchNorm2d(down_channels)
        self.bn2 = torch.nn.BatchNorm2d(output_channels)
        self.relu= torch.nn.ReLU()

    def forward(self, x):
        x1 = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x1 = self.sample(x1)
        x1 = self.bn2(x1)
        x = x + x1
        x = self.relu(x)
        return x

class Resnet(torch.nn.Module):
    def __init__(self, resnet, classes):
        super(Resnet, self).__init__()
        self.resnet_sign = resnet
        self.classes = classes
        self.conv1 = torch.nn.Conv2d(1, 16, 3, 1, 1)
        self.maxpool = torch.nn.MaxPool2d(3, 2, 1)
        self.averagepool = torch.nn.AvgPool2d(7,1,0)
        if self.resnet_sign == 18 or self.resnet_sign == 34:
            self.dense = torch.nn.Linear(32, self.classes)
        else:
            self.dense = torch.nn.Linear(2048, self.classes)
        self.set_blocks()


    def set_blocks(self):
        if self.resnet_sign == 18 or self.resnet_sign == 34:
            self.block1 = conv_block(16, 16)
            self.block2 = conv_block(16, 32)
            self.block3 = conv_block(32, 32)
        else:
            self.bottle_block1 = bottle_block(64, 64, 256)
            self.bottle_block2 = bottle_block(256, 64, 256)
            self.bottle_block3 = bottle_block(256, 128, 512, 2)
            self.bottle_block4 = bottle_block(512, 128, 512)
            self.bottle_block5 = bottle_block(512, 256, 1024, 2)
            self.bottle_block6 = bottle_block(1024, 256, 1024)
            self.bottle_block7 = bottle_block(1024, 512, 2048, 2)
            self.bottle_block8 = bottle_block(2048, 512, 2048)


    def forward(self, x):
        if self.resnet_sign == 18:
            bottle_sign = 0
            blocks = [2,2,2,2]
        elif self.resnet_sign == 34:
            bottle_sign = 0
            blocks = [3,4,6,3]
        elif self.resnet_sign == 50:
            bottle_sign = 1
            blocks = [3,4,6,3]
        elif self.resnet_sign == 101:
            bottle_sign = 1
            blocks = [3,4,23,3]
        elif self.resnet_sign == 152:
            bottle_sign = 1
            blocks = [3,8,36,3]
        else:
            raise ValueError("No such model")

        x = self.conv1(x)
        x = self.maxpool(x)
        for i in range(blocks[0]):
            if bottle_sign == 0:
                x = self.block1(x)
            else:
                if i == 0:
                    x = self.bottle_block1(x)
                else:
                    x = self.bottle_block2(x)
        for i in range(blocks[1]):
            if bottle_sign == 0:
                if i == 0:
                    x = self.block2(x)
                else:
                    x = self.block3(x)
            else:
                if i == 0:
                    x = self.bottle_block3(x)
                else:
                    x = self.bottle_block4(x)
        for i in range(blocks[2]):
            if bottle_sign == 1:
                if i == 0:
                    x = self.bottle_block5(x)
                else:
                    x = self.bottle_block6(x)
        for i in range(blocks[3]):
            if bottle_sign == 1:
                if i == 0:
                    x = self.bottle_block7(x)
                else:
                    x = self.bottle_block8(x)
        x = self.averagepool(x)
        x = torch.flatten(x, 1)
        x = self.dense(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# 用于ResNet50,101和152的残差块，用的是1x1+3x3+1x1的卷积
class Bottleneck(nn.Module):
    # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

class MNIST_optest(torch.nn.Module):
    def __init__(self):
        super(MNIST_optest, self).__init__()
        self.train_data = torchvision.datasets.MNIST(
            './mnist', train=True, transform=torchvision.transforms.ToTensor(), download=True
        )
        self.test_data = torchvision.datasets.MNIST(
            './mnist', train=False, transform=torchvision.transforms.ToTensor()
        )
        print("train_data:", self.train_data.train_data.size())
        print("train_labels:", self.train_data.train_labels.size())
        print("test_data:", self.test_data.test_data.size())

        self.train_loader = Data.DataLoader(dataset=self.train_data, batch_size=64, shuffle=True)
        self.test_loader = Data.DataLoader(dataset=self.test_data, batch_size=64)
        self.batch_size = 64

        self.net = ResNet18().cuda()
        self.old_net = ResNet18().cuda()
        parameters = list(self.net.parameters())
        old_parameters = list(self.old_net.parameters())
        for i in range(len(parameters)):
            old_parameters[i] = parameters[i].detach()
        summary(self.net, input_size=(1, 28, 28))

        self.beta = 0.9
        self.beta_correction = 1

        self.kp = 0
        self.ki = 1
        self.kd = 1

        self.grad_saving = []
        self.old_grad = []

        self.optimizer = torch.optim.SGD(self.net.parameters(), 0.005)
        self.old_optimizer = torch.optim.SGD(self.old_net.parameters(), 0.005)
        self.loss_func = torch.nn.CrossEntropyLoss().cuda()
        self.grad_saving = []

        for batch_x, batch_y in self.train_loader:
            batch_x, batch_y = Variable(batch_x).cuda(), Variable(batch_y).cuda()
            out = self.net(batch_x)
            loss = self.loss_func(out, batch_y)
            self.optimizer.zero_grad()
            loss.backward()
            a = list(self.net.parameters())
            for i in range(len(a)):
                self.grad_saving.append(a[i].grad/10000)
                self.old_grad.append(a[i].grad)
            break


    def momentum_optimizer(self, net, grad_saving):
        self.beta_correction *= self.beta
        model_parameters = list(net.parameters())
        for i in range(len(model_parameters)):
            grad1 = model_parameters[i].grad.detach()
            grad_saving[i] = (1-self.beta) * grad1 + self.beta * grad_saving[i]
            model_parameters[i].grad = grad_saving[i] / (1-self.beta_correction)
        self.optimizer.step()
        return net, grad_saving

    def PID_optimizer(self, net, grad_saving, batch_x, batch_y):
        old_out = self.old_net(batch_x)
        old_loss = self.loss_func(old_out, batch_y)
        old_parameters = list(self.old_net.parameters())
        self.old_optimizer.zero_grad()
        old_loss.backward()
        self.beta_correction *= self.beta
        model_parameters = list(net.parameters())
        for i in range(len(model_parameters)):
            grad1 = model_parameters[i].grad.detach()
            grad_saving[i] = (1-self.beta) * grad1 + self.beta * grad_saving[i]
            if self.dsign == 0:
                model_parameters[i].grad = self.ki * grad_saving[i] / (1-self.beta_correction)
                self.dsign = 1
            else:
                model_parameters[i].grad = self.ki * grad_saving[i] / (1-self.beta_correction) + self.kd * (grad_saving[i] - old_parameters[i].grad)
                old_parameters[i] = model_parameters[i].detach()
        self.old_optimizer.zero_grad()
        self.optimizer.step()

        return net, grad_saving




    def MNIST_training(self, optimizer_sign = 0):
        steps = 0
        episode = 0
        breaksign = 0
        self.dsign = 0
        losses = []
        accs = []
        for epoch in range(1000):
            print('epoch {}'.format(epoch + 1))
            # training-----------------------------

            train_loss = torch.tensor(0).float().cuda()
            train_acc = torch.tensor(0).float().cuda()



            for batch_x, batch_y in self.train_loader:
                batch_x, batch_y = Variable(batch_x).cuda(), Variable(batch_y).cuda()
                out = self.net(batch_x)
                loss = self.loss_func(out, batch_y)
                train_loss += loss.detach()
                pred = torch.max(out, dim=1)[1]
                train_correct = torch.sum(pred == batch_y)
                train_acc += train_correct
                self.optimizer.zero_grad()
                loss.backward()
                if optimizer_sign == 1:
                    self.net, self.grad_saving = self.momentum_optimizer(self.net, self.grad_saving)
                elif optimizer_sign == 2:
                    self.net, self.grad_saving = self.PID_optimizer(self.net, self.grad_saving, batch_x, batch_y)
                else:
                    self.optimizer.step()
                episode += 1
                if episode == 10:
                    episode = 0
                    steps += 1
                    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (10 * self.batch_size),
                                                                   train_acc / (10 * self.batch_size)))
                    train_loss = torch.tensor(0).float().cuda()
                    train_acc = torch.tensor(0).float().cuda()
                    # evaluation--------------------------------
                    self.net.eval()
                    eval_loss = torch.tensor(0).float().cuda()
                    eval_acc = torch.tensor(0).float().cuda()
                    with torch.no_grad():
                        for batch_x, batch_y in self.test_loader:
                            batch_x, batch_y = Variable(batch_x).cuda(), Variable(batch_y).cuda()
                            #plt.imshow(np.sum(batch_x[0].detach().cpu().numpy(), axis=0))
                            out = self.net(batch_x)
                            loss = self.loss_func(out, batch_y)
                            eval_loss += loss
                            pred = torch.max(out, dim=1)[1]
                            #print(pred)
                            num_correct = torch.sum(pred == batch_y)
                            eval_acc += num_correct.data
                    eval_loss /= len(self.test_data)
                    eval_acc /= len(self.test_data)
                    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss, eval_acc))
                    losses.append(eval_loss.detach().cpu().numpy())
                    accs.append(eval_acc.detach().cpu().numpy())
                    self.net.train()
                    if steps >= 20:
                        breaksign = 1
                        break
            if breaksign == 1:
                break

        return losses, accs, steps


losses = []
accs = []

for i in range(3):
    for j in range(2):
        a = MNIST_optest()
        loss, acc, step = a.MNIST_training(i)
        if j == 0 :
            lossi = np.array(loss)
            acci = np.array(acc)
        else:
            lossi += np.array(loss)
            acci += np.array(acc)

    losses.append(lossi/5)
    accs.append(acci/5)

x = range(0,20)
for i in range(3):
    plt.plot(x,accs[i],label='loss' + str(i))
plt.show()
for i in range(3):
    plt.plot(x,losses[i],label='loss' + str(i))
plt.show()



