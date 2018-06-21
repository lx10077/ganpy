import argparse
import os
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from wgan.wgan_nets import Generator, Discriminator

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


# Initialize generator and discriminator
img_shape = (args.channels, args.img_size, args.img_size)
generator = Generator(args.latent_dim, img_shape)
discriminator = Discriminator(img_shape)

if use_cuda:
    generator.cuda()
    discriminator.cuda()

# Configure data loader
os.makedirs('../../data/mnist', exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('../../data/mnist', train=True, download=True,
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
        loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))

        DiscrimUpdater.zero_grad()
        loss_D.backward()
        DiscrimUpdater.step()

        # Clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-args.clip_value, args.clip_value)

        # Train the generator every n_critic iterations
        if i % args.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            gen_imgs = generator(z)
            # Adversarial loss
            loss_G = -torch.mean(discriminator(gen_imgs))

            GenUpdater.zero_grad()
            loss_G.backward()
            GenUpdater.step()

            print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, args.n_epochs,
                                                                              batches_done % len(dataloader), len(dataloader),
                                                                              loss_D.item(), loss_G.item()))

        if batches_done % args.sample_interval == 0:
            save_image(gen_imgs.data[:25], 'images/%d.png' % batches_done, nrow=5, normalize=True)
        batches_done += 1
