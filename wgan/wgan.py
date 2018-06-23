import argparse
import os
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch


parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--n_critic', type=int, default=5, help='number of training steps for discriminator per iter')
parser.add_argument('--clip_value', type=float, default=0.01, help='lower and upper clip value for disc. weights')
parser.add_argument('--sample_interval', type=int, default=400, help='interval betwen image samples')
args = parser.parse_args()
os.makedirs('images', exist_ok=True)
use_cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        assert len(img_shape) == 3
        self.img_shape = img_shape
        self.latent_dim = latent_dim

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, x):
        img = self.model(x)
        img = img.view(-1, self.img_shape[0], self.img_shape[1], self.img_shape[2])
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        assert len(img_shape) == 3
        self.img_shape = img_shape

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


# Initialize generator and discriminator
img_shape = (args.channels, args.img_size, args.img_size)
generator = Generator(args.latent_dim, img_shape)
discriminator = Discriminator(img_shape)

if use_cuda:
    generator.cuda()
    discriminator.cuda()

# Configure data loader
os.makedirs('../data/mnist', exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('../data/mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=args.batch_size, shuffle=True)

# Define optimizers for generator and discriminator
GenUpdater = torch.optim.RMSprop(generator.parameters(), lr=args.lr)
DiscrimUpdater = torch.optim.RMSprop(discriminator.parameters(), lr=args.lr)

# Use cuda if possible
Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

batches_done = 0
for epoch in range(args.n_epochs):

    for i, (imgs, _) in enumerate(dataloader):

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))))

        # Generate a batch of images
        fake_imgs = generator(z).detach()

        # Adversarial loss
        d_loss = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))

        DiscrimUpdater.zero_grad()
        d_loss.backward()
        DiscrimUpdater.step()

        # Clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-args.clip_value, args.clip_value)

        # Train the generator every n_critic iterations
        if i % args.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------
            
            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))))

            # Generate a batch of images
            gen_imgs = generator(z)

            # Adversarial loss
            g_loss = -torch.mean(discriminator(gen_imgs))

            GenUpdater.zero_grad()
            g_loss.backward()
            GenUpdater.step()

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (
                epoch, args.n_epochs, i, len(dataloader), d_loss.data[0], g_loss.data[0]))

        if batches_done % args.sample_interval == 0:
            save_image(gen_imgs.data[:25], 'images/%d.png' % batches_done, nrow=5, normalize=True)
        batches_done += 1
