import argparse
import torch
import os
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets


parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--k', type=int, default=3, help='')
parser.add_argument('--gpu', action="store_true", default=False, help='use gpu')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=400, help='interval betwen image samples')
args = parser.parse_args()
os.makedirs('./images', exist_ok=True)
use_cuda = True if torch.cuda.is_available() and args.gpu else False

# Loss function
adversarial_loss = torch.nn.BCELoss()


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
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    adversarial_loss = adversarial_loss.cuda()


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
dataloader = data_loader(args.batch_size * args.k)

# Define optimizers for generator and discriminator
GenUpdater = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
DiscrimUpdater = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

# Use cuda if possible
Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


def main(args):
    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # Adversarial ground truths
            valid = Variable(Tensor(args.batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(args.batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input, k * batch zise images in total
            real_imgs_tensor = imgs.type(Tensor)

            # Prepare at most k batches of real images
            k_real_imgs = []
            for k in range(int(len(real_imgs_tensor) / args.batch_size)):
                k_real_imgs.append(Variable(real_imgs_tensor[k * args.batch_size: (k + 1) * args.batch_size]))

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Update discriminator k times
            for real_imgs in k_real_imgs:

                # Sample noise as generator input, then generate a batch of images
                z = Variable(Tensor(np.random.normal(0, 1, (args.batch_size, args.latent_dim))))
                gen_imgs = generator(z)

                # Measure discriminator's ability to classify real from generated samples
                real_loss = adversarial_loss(discriminator(real_imgs), valid)
                fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                # Update the discriminator by ascending its stochastic gradient
                DiscrimUpdater.zero_grad()
                d_loss.backward()
                DiscrimUpdater.step()

            # -----------------
            #  Train Generator
            # -----------------

            # Loss measures generator's ability to fool the discriminator
            gen_imgs = generator(Variable(Tensor(np.random.normal(0, 1, (args.batch_size, args.latent_dim)))))
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            # Update generator
            GenUpdater.zero_grad()
            g_loss.backward()
            GenUpdater.step()

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (
                epoch, args.n_epochs, i, len(dataloader), d_loss.data[0], g_loss.data[0]))

            batches_done = epoch * len(dataloader) + i
            if batches_done % args.sample_interval == 0:
                save_image(gen_imgs.data[:25], './images/%d.png' % batches_done, nrow=5, normalize=True)


if __name__ == '__main__':
    main(args)
