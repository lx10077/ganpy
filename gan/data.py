import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import torch
import os


# Make data dir if data_dir doesn't exist
os.makedirs('../data/mnist', exist_ok=True)


# Configure data loader
def data_loader(batch_size):
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('../data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=batch_size, shuffle=True)
    return dataloader
