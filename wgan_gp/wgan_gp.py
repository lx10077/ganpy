import argparse
import os
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn as nn
import torch.autograd as autograd
import torch


parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--n_critic', type=int, default=5, help='number of training steps for discriminator per iter')
parser.add_argument('--clip_value', type=float, default=0.01, help='lower and upper clip value for disc. weights')
parser.add_argument('--sample_interval', type=int, default=800, help='interval betwen image samples')
args = parser.parse_args()

os.makedirs('images', exist_ok=True)
use_cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(args.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


# Initialize generator and discriminator
img_shape = (args.channels, args.img_size, args.img_size)
generator = Generator()
discriminator = Discriminator()

# Loss weight for gradient penalty
lambda_gp = 10

if use_cuda:
    generator.cuda()
    discriminator.cuda()

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


# Set dataloader
dataloader = data_loader(args.batch_size)

# Define optimizers for generator and discriminator
GenUpdater = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
DiscrimUpdater = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

# Use cuda if possible
Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""

    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))

    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Tensor(real_samples.shape[0], 1).fill_(1.0)

    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates,
                              grad_outputs=fake, create_graph=True, retain_graph=True,
                              only_inputs=True)[0]
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


batches_done = 0
for epoch in range(args.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Configure input
        real_imgs = imgs.type(Tensor)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Sample noise as generator input
        z = Tensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim)))

        # Generate a batch of images
        fake_imgs = generator(z)

        # Real images
        real_validity = discriminator(real_imgs)

        # Fake images
        fake_validity = discriminator(fake_imgs)

        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)

        # Adversarial loss
        d_loss = fake_validity.mean() - real_validity.mean() + lambda_gp * gradient_penalty

        DiscrimUpdater.zero_grad()
        d_loss.backward()
        DiscrimUpdater.step()

        GenUpdater.zero_grad()

        # Train the generator every n_critic steps
        if i % args.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_imgs = generator(z)

            # Train on fake images
            fake_validity = discriminator(fake_imgs)

            # Loss measures generator's ability to fool the discriminator
            g_loss = - fake_validity.mean()

            g_loss.backward()
            GenUpdater.step()

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (
                epoch, args.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item()))

            if batches_done % args.sample_interval == 0:
                save_image(fake_imgs.data[:25], 'images/%d.png' % batches_done, nrow=5, normalize=True)

            batches_done += args.n_critic
