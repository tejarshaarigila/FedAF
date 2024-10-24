# utils/utils.py

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader, TensorDataset
from torchvision import datasets, transforms
from scipy.stats import wasserstein_distance
import logging
import time

# networks
''' Swish activation '''
class Swish(nn.Module): # Swish(x) = x∗σ(x)
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)


''' MLP '''
class MLP(nn.Module):
    def __init__(self, channel, num_classes):
        super(MLP, self).__init__()
        self.fc_1 = nn.Linear(28*28*1 if channel==1 else 32*32*3, 128)
        self.fc_2 = nn.Linear(128, 128)
        self.fc_3 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = F.relu(self.fc_1(out))
        out = F.relu(self.fc_2(out))
        out = self.fc_3(out)
        return out



''' ConvNet '''
class ConvNet(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32)):
        super(ConvNet, self).__init__()

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def embed(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        elif net_act == 'swish':
            return Swish()
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat



''' LeNet '''
class LeNet(nn.Module):
    def __init__(self, channel, num_classes):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel, 6, kernel_size=5, padding=2 if channel==1 else 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_1 = nn.Linear(16 * 5 * 5, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = self.fc_3(x)
        return x



''' AlexNet '''
class AlexNet(nn.Module):
    def __init__(self, channel, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=5, stride=1, padding=4 if channel==1 else 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(192 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def embed(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


''' AlexNetBN '''
class AlexNetBN(nn.Module):
    def __init__(self, channel, num_classes):
        super(AlexNetBN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=5, stride=1, padding=4 if channel==1 else 2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(192 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def embed(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


''' VGG '''
cfg_vgg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
class VGG(nn.Module):
    def __init__(self, vgg_name, channel, num_classes, norm='instancenorm'):
        super(VGG, self).__init__()
        self.channel = channel
        self.features = self._make_layers(cfg_vgg[vgg_name], norm)
        self.classifier = nn.Linear(512 if vgg_name != 'VGGS' else 128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def embed(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

    def _make_layers(self, cfg, norm):
        layers = []
        in_channels = self.channel
        for ic, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=3 if self.channel==1 and ic==0 else 1),
                           nn.GroupNorm(x, x, affine=True) if norm=='instancenorm' else nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def VGG11(channel, num_classes):
    return VGG('VGG11', channel, num_classes)
def VGG11BN(channel, num_classes):
    return VGG('VGG11', channel, num_classes, norm='batchnorm')
def VGG13(channel, num_classes):
    return VGG('VGG13', channel, num_classes)
def VGG16(channel, num_classes):
    return VGG('VGG16', channel, num_classes)
def VGG19(channel, num_classes):
    return VGG('VGG19', channel, num_classes)


''' ResNet_AP '''
# The conv(stride=2) is replaced by conv(stride=1) + avgpool(kernel_size=2, stride=2)

class BasicBlock_AP(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(BasicBlock_AP, self).__init__()
        self.norm = norm
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False) # modification
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=1, bias=False),
                nn.AvgPool2d(kernel_size=2, stride=2), # modification
                nn.GroupNorm(self.expansion * planes, self.expansion * planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.stride != 1: # modification
            out = F.avg_pool2d(out, kernel_size=2, stride=2)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck_AP(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(Bottleneck_AP, self).__init__()
        self.norm = norm
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False) # modification
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(self.expansion * planes, self.expansion * planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=1, bias=False),
                nn.AvgPool2d(kernel_size=2, stride=2),  # modification
                nn.GroupNorm(self.expansion * planes, self.expansion * planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        if self.stride != 1: # modification
            out = F.avg_pool2d(out, kernel_size=2, stride=2)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_AP(nn.Module):
    def __init__(self, block, num_blocks, channel=3, num_classes=10, norm='instancenorm'):
        super(ResNet_AP, self).__init__()
        self.in_planes = 64
        self.norm = norm

        self.conv1 = nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(64, 64, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = nn.Linear(512 * block.expansion * 3 * 3 if channel==1 else 512 * block.expansion * 4 * 4, num_classes)  # modification

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, kernel_size=1, stride=1) # modification
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def embed(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, kernel_size=1, stride=1) # modification
        out = out.view(out.size(0), -1)
        return out

def ResNet18BN_AP(channel, num_classes):
    return ResNet_AP(BasicBlock_AP, [2,2,2,2], channel=channel, num_classes=num_classes, norm='batchnorm')

def ResNet18_AP(channel, num_classes):
    return ResNet_AP(BasicBlock_AP, [2,2,2,2], channel=channel, num_classes=num_classes)


''' ResNet '''

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(BasicBlock, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(self.expansion*planes, self.expansion*planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(Bottleneck, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(self.expansion*planes, self.expansion*planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(self.expansion*planes, self.expansion*planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, channel=3, num_classes=10, norm='instancenorm'):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.norm = norm

        self.conv1 = nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(64, 64, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm))
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
        out = self.classifier(out)
        return out

    def embed(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out


def ResNet18BN(channel, num_classes):
    return ResNet(BasicBlock, [2,2,2,2], channel=channel, num_classes=num_classes, norm='batchnorm')

def ResNet18(channel, num_classes):
    return ResNet(BasicBlock, [2,2,2,2], channel=channel, num_classes=num_classes)

def ResNet34(channel, num_classes):
    return ResNet(BasicBlock, [3,4,6,3], channel=channel, num_classes=num_classes)

def ResNet50(channel, num_classes):
    return ResNet(Bottleneck, [3,4,6,3], channel=channel, num_classes=num_classes)

def ResNet101(channel, num_classes):
    return ResNet(Bottleneck, [3,4,23,3], channel=channel, num_classes=num_classes)

def ResNet152(channel, num_classes):
    return ResNet(Bottleneck, [3,8,36,3], channel=channel, num_classes=num_classes)

# Configure logging for utilities
logger = logging.getLogger('Common.Utils')
if not logger.handlers:
    log_directory = "/home/t914a431/log"
    os.makedirs(log_directory, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_directory, 'utils.log'))
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

def load_data(dataset_name, data_path='data', train=True):
    """Load the dataset and apply necessary transformations."""
    logger.info("Loading dataset: %s", dataset_name)
    if dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                 std=(0.2023, 0.1994, 0.2010))
        ])
        dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    elif dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    elif dataset_name == 'CelebA':
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        dataset = datasets.CelebA(root=data_path, split='train', download=True, transform=transform)
    else:
        logger.error(f"Unsupported dataset: {dataset_name}")
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    logger.info("Dataset loaded successfully.")
    return dataset

def partition_data_unique_rounds(dataset, num_clients, num_rounds, alpha, seed=42):
    """
    Partition the dataset into unique, non-overlapping subsets for each client across all rounds,
    ensuring no data is reused in subsequent rounds.
    """
    logger = logging.getLogger('Common.Utils')
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info("Starting dataset partitioning across all rounds and clients with Dirichlet distribution.")

    # Extract labels and create indices for each class
    labels = np.array(dataset.targets)
    num_classes = np.max(labels) + 1
    label_indices = [np.where(labels == i)[0].tolist() for i in range(num_classes)]

    # Shuffle the indices for each class
    for c in range(num_classes):
        np.random.shuffle(label_indices[c])

    # Calculate total number of allocations
    total_allocations = num_rounds * num_clients

    # Initialize client_indices_per_round as a list
    client_indices_per_round = [[] for _ in range(num_rounds)]
    for round_num in range(num_rounds):
        client_indices_per_round[round_num] = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        available_indices = label_indices[c]
        total_samples = len(available_indices)
        if total_samples < total_allocations:
            logger.warning(f"Not enough data for class {c}. Needed: {total_allocations}, Available: {total_samples}")
            # Optionally, handle this scenario as needed
            allocations = np.random.dirichlet([alpha] * total_allocations) * total_samples
            allocations = allocations.astype(int)
            allocations[-1] = total_samples - allocations[:-1].sum()  # Adjust last allocation
        else:
            # Allocate without exceeding available samples
            allocations = np.random.dirichlet([alpha] * total_allocations) * total_samples
            allocations = allocations.astype(int)
            allocations[-1] = total_samples - allocations[:-1].sum()  # Adjust last allocation

        # Assign allocations to each round and client
        start = 0
        for round_num in range(num_rounds):
            for client_id in range(num_clients):
                allocation_index = round_num * num_clients + client_id
                if allocation_index >= len(allocations):
                    num_samples = 0
                else:
                    num_samples = allocations[allocation_index]
                end = start + num_samples
                if end > total_samples:
                    end = total_samples
                client_indices_per_round[round_num][client_id].extend(available_indices[start:end])
                start = end
                if start >= total_samples:
                    break
            if start >= total_samples:
                break

    return client_indices_per_round

    for c in range(num_classes):
        available_indices = label_indices[c]
        total_samples = len(available_indices)
        if total_samples < total_allocations:
            logger.warning(f"Not enough data for class {c}. Needed: {total_allocations}, Available: {total_samples}")
            # Optionally, you can decide to allow some reuse or handle this differently
            # For now, we'll proceed by allocating as much as possible
            allocations = np.random.dirichlet([alpha] * total_allocations) * total_samples
            allocations = allocations.astype(int)
            allocations[-1] = total_samples - allocations[:-1].sum()  # Adjust last allocation
        else:
            # Allocate without exceeding available samples
            allocations = np.random.dirichlet([alpha] * total_allocations) * total_samples
            allocations = allocations.astype(int)
            allocations[-1] = total_samples - allocations[:-1].sum()  # Adjust last allocation

        # Assign allocations to each round and client
        start = 0
        for round_num in range(num_rounds):
            for client_id in range(num_clients):
                num_samples = allocations[round_num * num_clients + client_id]
                end = start + num_samples
                if end > total_samples:
                    end = total_samples
                client_indices_per_round[round_num][client_id].extend(available_indices[start:end])
                start = end
                if start >= total_samples:
                    break
            if start >= total_samples:
                break

    logger.info("Dataset partitioning across all rounds and clients completed successfully.")
    return client_indices_per_round

def save_partitions(client_indices_per_round, save_dir):
    """
    Save the client indices for each round and each client.

    Args:
        client_indices_per_round (list): A list where each element corresponds to a round and contains a list of lists of client indices.
        save_dir (str): Directory where partitions are to be saved.
    """
    logger = logging.getLogger('GeneratePartitions')
    for round_num, client_indices in enumerate(client_indices_per_round):
        round_dir = os.path.join(save_dir, f'round_{round_num}')
        os.makedirs(round_dir, exist_ok=True)
        for client_id, indices in enumerate(client_indices):
            partition_path = os.path.join(round_dir, f'client_{client_id}_partition.pkl')
            try:
                with open(partition_path, 'wb') as f:
                    pickle.dump(indices, f)
                logger.info(f"Saved partition for Client {client_id} in Round {round_num} at {partition_path}")
            except Exception as e:
                logger.error(f"Failed to save partition for Client {client_id} in Round {round_num}: {e}")

import os
import torch
import pickle
from torch.utils.data import Subset
import logging

def load_partitions(dataset, num_clients, num_rounds, partition_dir, dataset_name, model_name, honesty_ratio):
    """
    Load pre-partitioned data for each client for each round.

    Args:
        dataset (torch.utils.data.Dataset): The original dataset.
        num_clients (int): Number of clients.
        num_rounds (int): Number of communication rounds.
        partition_dir (str): Directory where partitions are saved.
        dataset_name (str): Name of the dataset (e.g., CIFAR10, MNIST).
        model_name (str): Model used for the partitioning (e.g., ConvNet, ResNet).
        honesty_ratio (str): Honesty ratio, as part of the directory structure.

    Returns:
        dict: A dictionary with keys as round numbers and values as lists of Subset datasets for each client.
              If a partition is missing, the client will have an empty dataset for that round.
    """
    client_datasets_per_round = {}
    logger = logging.getLogger('FedAF.Main')  # Ensure logger is obtained correctly

    for round_num in range(num_rounds):
        # Navigate to the round-specific directory
        round_dir = os.path.join(partition_dir, f'round_{round_num}')
        client_datasets = []
        for client_id in range(num_clients):
            partition_path = os.path.join(round_dir, f'client_{client_id}_partition.pkl')
            
            # Log the exact path being checked
            logger.info(f"Checking partition path: {partition_path}")

            if os.path.exists(partition_path):
                try:
                    with open(partition_path, 'rb') as f:
                        indices = pickle.load(f)
                    logger.debug(f"Client {client_id} for round {round_num} has {len(indices)} data points.")
                    client_subset = Subset(dataset, indices)
                    client_datasets.append(client_subset)
                except Exception as e:
                    logger.error(f"Error loading partition for Client {client_id} in Round {round_num}: {e}")
                    client_datasets.append(Subset(dataset, []))  # Append empty Subset on error
            else:
                # Append empty Subset if partition file is missing
                logger.warning(f"Round {round_num}, Client {client_id}: Partition file missing. Assigning empty dataset.")
                client_datasets.append(Subset(dataset, []))
                    
        client_datasets_per_round[round_num] = client_datasets

    logger.info("All data partitions loaded successfully.")
    return client_datasets_per_round

def get_network(model_name, channel, num_classes, im_size=(32, 32), device='cpu'):
    """Initializes the network based on the model name."""
    torch.random.manual_seed(int(time.time() * 1000) % 100000)
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()

    if model_name == 'MLP':
        net = MLP(channel=channel, num_classes=num_classes)
    elif model_name == 'ConvNet':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                     net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model_name == 'LeNet':
        net = LeNet(channel=channel, num_classes=num_classes)
    elif model_name == 'AlexNet':
        net = AlexNet(channel=channel, num_classes=num_classes)
    elif model_name == 'AlexNetBN':
        net = AlexNetBN(channel=channel, num_classes=num_classes)
    elif model_name == 'VGG11':
        net = VGG11(channel=channel, num_classes=num_classes)
    elif model_name == 'VGG11BN':
        net = VGG11BN(channel=channel, num_classes=num_classes)
    elif model_name == 'ResNet18':
        net = ResNet18(channel=channel, num_classes=num_classes)
    elif model_name == 'ResNet18BN_AP':
        net = ResNet18BN_AP(channel=channel, num_classes=num_classes)
    elif model_name == 'ResNet18BN':
        net = ResNet18BN(channel=channel, num_classes=num_classes)
    else:
        logger.error(f"Unsupported model '{model_name}'.")
        raise ValueError(f"Unsupported model '{model_name}'.")
    
    net = net.to(device)
    logger.info(f"Initialized model '{model_name}' on device '{device}'.")
    return net

def get_default_convnet_setting():
    """Provides default settings for the ConvNet architecture."""
    net_width = 128
    net_depth = 3
    net_act = 'relu'
    net_norm = 'instancenorm'
    net_pooling = 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling


def compute_swd(logits1, logits2, num_projections=100):
    """
    Computes the Sliced Wasserstein Distance (SWD) between two sets of logits.

    Args:
        logits1 (torch.Tensor or np.ndarray): First set of logits.
        logits2 (torch.Tensor or np.ndarray): Second set of logits.
        num_projections (int): Number of random projections.

    Returns:
        float: Average SWD over all projections.
    """
    if isinstance(logits1, torch.Tensor):
        logits1 = logits1.detach().cpu().numpy()
    if isinstance(logits2, torch.Tensor):
        logits2 = logits2.detach().cpu().numpy()

    # Check if logits are valid arrays
    if logits1.ndim == 0 or logits2.ndim == 0:
        return 0.0  # Return zero distance if logits are scalars or zero-dimensional

    dimensions = logits1.shape[0]
    if dimensions == 0:
        return 0.0  # Return zero distance if dimension is zero

    # Generate projections
    projections = np.random.normal(0, 1, (num_projections, dimensions))
    projections /= np.linalg.norm(projections, axis=1, keepdims=True)

    # Project the logits
    projected_logits1 = projections.dot(logits1)
    projected_logits2 = projections.dot(logits2)

    # Compute Wasserstein distance for each projection
    distances = []
    for i in range(num_projections):
        distance = wasserstein_distance([projected_logits1[i]], [projected_logits2[i]])
        distances.append(distance)
    average_distance = np.mean(distances)
    return average_distance


def calculate_logits_labels(model_net, partition, num_classes, device, path, ipc, temperature):
    """
    Calculates and saves class-wise averaged logits (V(k,c)) and probabilities (R(k,c)).

    Args:
        model_net (torch.nn.Module): The global model.
        partition (Subset): Client's data partition.
        num_classes (int): Number of classes.
        device (torch.device): Device to perform computations on.
        path (str): Directory path to save logits.
        ipc (int): Instances per class.
        temperature (float): Temperature parameter for softmax.
    """

    # Create subdirectories if they don't exist
    os.makedirs(path, exist_ok=True)

    # Create DataLoader for the client's partition
    dataloader = DataLoader(partition, batch_size=256, shuffle=False)

    # Initialize storage for logits and probabilities
    logits_by_class = [torch.empty((0, num_classes), device=device) for _ in range(num_classes)]
    probs_by_class = [torch.empty((0, num_classes), device=device) for _ in range(num_classes)]

    model_net.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model_net(images)
            probs = F.softmax(logits / temperature, dim=1)
            for i in range(labels.size(0)):
                label = labels[i].item()
                if 0 <= label < num_classes:
                    logits_by_class[label] = torch.cat((logits_by_class[label], logits[i].unsqueeze(0)), dim=0)
                    probs_by_class[label] = torch.cat((probs_by_class[label], probs[i].unsqueeze(0)), dim=0)
                else:
                    logger.warning(f"calculate_logits_labels: Label {label} is out of range. Skipping.")

    # Average logits and probabilities per class
    logits_avg = []
    probs_avg = []
    for c in range(num_classes):
        if logits_by_class[c].size(0) >= ipc:
            avg_logit = logits_by_class[c].mean(dim=0)
            avg_prob = probs_by_class[c].mean(dim=0)
            logger.debug(f"calculate_logits_labels: Class {c} - Avg Logit: {avg_logit}, Avg Prob: {avg_prob}")
        else:
            avg_logit = torch.zeros(num_classes, device=device)
            avg_prob = torch.zeros(num_classes, device=device)
            logger.warning(f"calculate_logits_labels: Not enough instances for class {c}. Initialized avg_logit and avg_prob with zeros.")
        logits_avg.append(avg_logit)
        probs_avg.append(avg_prob)

    # Save the averaged logits and probabilities
    try:
        for c in range(num_classes):
            torch.save(logits_avg[c], os.path.join(path, f'Vkc_{c}.pt'))
            torch.save(probs_avg[c], os.path.join(path, f'Rkc_{c}.pt'))
        logger.info(f"calculate_logits_labels: Saved averaged logits and probabilities to {path}.")
    except Exception as e:
        logger.error(f"calculate_logits_labels: Error saving logits and probabilities - {e}")


def load_latest_model(model_dir, model_name, channel, num_classes, im_size, device):
    """
    Loads the latest global model from the model directory.

    Args:
        model_dir (str): Directory containing model checkpoints.
        model_name (str): Name of the model architecture.
        channel (int): Number of input channels.
        num_classes (int): Number of output classes.
        im_size (tuple): Image size (height, width).
        device (str): Device to load the model on.

    Returns:
        torch.nn.Module: Loaded model.
    """
    try:
        if os.path.exists(model_dir) and os.listdir(model_dir):
            model_files = [
                os.path.join(model_dir, f) for f in os.listdir(model_dir)
                if f.endswith('.pth') and f.startswith('fedaf')
            ]
            if model_files:
                latest_model_file = max(model_files, key=os.path.getmtime)
                model = get_network(
                    model_name=model_name,
                    channel=channel,
                    num_classes=num_classes,
                    im_size=im_size,
                    device=device
                )
                state_dict = torch.load(latest_model_file, map_location=device)
                model.load_state_dict(state_dict)
                logger.info(f"load_latest_model: Loaded model from {latest_model_file}.")
                return model
        # If no model exists, initialize a new one
        logger.info("load_latest_model: Model directory is empty or no valid model found. Initializing a new model.")
        model = get_network(
            model_name=model_name,
            channel=channel,
            num_classes=num_classes,
            im_size=im_size,
            device=device
        )
        return model
    except Exception as e:
        logger.error(f"load_latest_model: Error loading the latest model: {e}")
        # Initialize a new model in case of error
        model = get_network(
            model_name=model_name,
            channel=channel,
            num_classes=num_classes,
            im_size=im_size,
            device=device
        )
        return model


def compute_T(model, synthetic_dataset, num_classes, temperature, device):
    """
    Computes the class-wise averaged soft labels T from the model's predictions on the synthetic data.

    Args:
        model (torch.nn.Module): The global model.
        synthetic_dataset (TensorDataset): Synthetic data dataset.
        num_classes (int): Number of classes.
        temperature (float): Temperature parameter for softmax scaling.
        device (str): Device to perform computations on.

    Returns:
        torch.Tensor: Tensor of class-wise averaged soft labels T.
    """
    model.eval()
    class_logits_sum = [torch.zeros(num_classes, device=device) for _ in range(num_classes)]
    class_counts = [0 for _ in range(num_classes)]

    synthetic_loader = DataLoader(synthetic_dataset, batch_size=256, shuffle=False)

    with torch.no_grad():
        for inputs, labels in synthetic_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # [batch_size, num_classes]
            for i in range(inputs.size(0)):
                label = labels[i].item()
                if 0 <= label < num_classes:
                    class_logits_sum[label] += outputs[i]
                    class_counts[label] += 1

    t_list = []
    for c in range(num_classes):
        if class_counts[c] > 0:
            avg_logit = class_logits_sum[c] / class_counts[c]
            t_c = nn.functional.softmax(avg_logit / temperature, dim=0)
            t_list.append(t_c)
        else:
            # Initialize with uniform distribution if no data for class c
            t_list.append(torch.ones(num_classes, device=device) / num_classes)
            logger.warning(f"compute_T: No synthetic data for class {c}. Initialized T with uniform distribution.")

    t_tensor = torch.stack(t_list)  # [num_classes, num_classes]
    logger.info("compute_T: Computed class-wise averaged soft labels T.")
    return t_tensor


def get_eval_pool(eval_mode, model, model_eval):
    """
    Prepares a pool of models for evaluation based on the evaluation mode.

    Args:
        eval_mode (str): Evaluation mode ('S', 'SS', etc.).
        model (str): Current model architecture.
        model_eval (str): Model architecture for evaluation.

    Returns:
        list: List containing model architectures for evaluation.
    """
    if eval_mode == 'S':  # Self
        if 'BN' in model:
            logger.warning('get_eval_pool: Replacing BatchNorm with InstanceNorm in evaluation.')
        try:
            bn_index = model.index('BN')
            model_eval_pool = [model[:bn_index]]
        except ValueError:
            model_eval_pool = [model]
    elif eval_mode == 'SS':  # Self-Self
        model_eval_pool = [model]
    else:
        model_eval_pool = [model_eval]
    return model_eval_pool


def plot_accuracies(test_accuracies, model_name, dataset, alpha, num_partitions, num_clients=None, save_dir='plots'):
    """
    Plots and saves the test accuracies over communication rounds.

    Args:
        test_accuracies (list): List of test accuracies per round.
        model_name (str): Name of the model used.
        dataset (str): Name of the dataset used.
        alpha (float): Dirichlet distribution parameter.
        num_partitions (int): Number of client partitions.
        num_clients (int, optional): Number of clients (FedAvg only).
        save_dir (str): Directory to save the plot.
    """
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)
    rounds = range(1, len(test_accuracies) + 1)
    plt.figure()
    plt.plot(rounds, test_accuracies, marker='o')
    plt.title(f"Test Accuracy over Rounds\nModel: {model_name}, Dataset: {dataset}, Alpha: {alpha}")
    plt.xlabel("Round")
    plt.ylabel("Test Accuracy (%)")
    plt.grid(True)
    if num_clients:
        plt.legend([f"{num_clients} Clients"])
    save_path = os.path.join(save_dir, f"accuracy_{model_name}_{dataset}_alpha{alpha}.png")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"plot_accuracies: Test accuracy graph saved to {save_path}.")
