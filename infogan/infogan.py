import argparse
import os
import numpy as np
import itertools
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn as nn
import torch


parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=62, help='dimensionality of the latent space')
parser.add_argument('--code_dim', type=int, default=2, help='latent code')
parser.add_argument('--n_classes', type=int, default=10, help='number of classes for dataset')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=4000, help='interval between image sampling')
args = parser.parse_args()

os.makedirs('images/static/', exist_ok=True)
os.makedirs('images/varying_c1/', exist_ok=True)
os.makedirs('images/varying_c2/', exist_ok=True)
use_cuda = True if torch.cuda.is_available() else False


class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.align_corners = align_corners
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.interp(input=x,
                        size=self.size,
                        scale_factor=self.scale_factor,
                        mode=self.mode,
                        align_corners=self.align_corners)
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        input_dim = args.latent_dim + args.n_classes + args.code_dim

        self.init_size = args.img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(input_dim, 128*self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            Interpolate(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            Interpolate(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, args.channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels, code):
        gen_input = torch.cat((noise, labels, code), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(args.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = args.img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128*ds_size**2, 1))
        self.aux_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, args.n_classes),
            nn.Softmax(dim=1)
        )
        self.latent_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, args.code_dim))

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        # Q produces latent codes and shares many layers with discriminator.
        label = self.aux_layer(out)
        latent_code = self.latent_layer(out)

        return validity, label, latent_code


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def to_categorical(y, num_columns):
    """Returns one-hot encoded Variable"""

    y_cat = np.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.

    return FloatTensor(y_cat)


# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Loss functions
adversarial_loss = torch.nn.MSELoss()
categorical_loss = torch.nn.CrossEntropyLoss()
continuous_loss = torch.nn.MSELoss()

# Loss weights
lambda_cat = 1
lambda_con = 0.1

if use_cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    categorical_loss.cuda()
    continuous_loss.cuda()


# Configure data loader
os.makedirs('../data/mnist', exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('../data/mnist', train=True, download=True,
                   transform=transforms.Compose([
                        transforms.Resize(args.img_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=args.batch_size, shuffle=True, drop_last=True)


# Define optimizers for generator, discriminator and Q
GenUpdater = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
DiscrimUpdater = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
InfoUpdater = torch.optim.Adam(itertools.chain(generator.parameters(), discriminator.parameters()),
                               lr=args.lr, betas=(args.b1, args.b2))


# Reset the input tensor type if use cuda
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor


# Static generator inputs for sampling
static_z = FloatTensor(np.zeros((args.n_classes ** 2, args.latent_dim)))
static_label = to_categorical(np.array([num for _ in range(args.n_classes) for num in range(args.n_classes)]),
                              num_columns=args.n_classes)
static_code = FloatTensor(np.zeros((args.n_classes ** 2, args.code_dim)))


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""

    # Static sample
    z = FloatTensor(np.random.normal(0, 1, (n_row ** 2, args.latent_dim)))
    static_sample = generator(z, static_label, static_code)
    save_image(static_sample.data, 'images/static/%d.png' % batches_done, nrow=n_row, normalize=True)

    # Get varied c1 and c2
    zeros = np.zeros((n_row**2, 1))
    c_varied = np.repeat(np.linspace(-1, 1, n_row)[:, np.newaxis], n_row, 0)
    c1 = FloatTensor(np.concatenate((c_varied, zeros), -1))
    c2 = FloatTensor(np.concatenate((zeros, c_varied), -1))
    sample1 = generator(static_z, static_label, c1)
    sample2 = generator(static_z, static_label, c2)
    save_image(sample1.data, 'images/varying_c1/%d.png' % batches_done, nrow=n_row, normalize=True)
    save_image(sample2.data, 'images/varying_c2/%d.png' % batches_done, nrow=n_row, normalize=True)


for epoch in range(args.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = FloatTensor(batch_size, 1).fill_(1.0)
        fake = FloatTensor(batch_size, 1).fill_(0.0)

        # Configure input
        real_imgs = imgs.type(FloatTensor)
        labels = to_categorical(labels.numpy(), num_columns=args.n_classes)

        # -----------------
        #  Train Generator
        # -----------------

        # Sample noise and labels as generator input
        z = FloatTensor(np.random.normal(0, 1, (batch_size, args.latent_dim)))
        label_input = to_categorical(np.random.randint(0, args.n_classes, batch_size), num_columns=args.n_classes)
        code_input = FloatTensor(np.random.uniform(-1, 1, (batch_size, args.code_dim)))

        # Generate a batch of images
        gen_imgs = generator(z, label_input, code_input)

        # Loss measures generator's ability to fool the discriminator
        validity, _, _ = discriminator(gen_imgs)
        g_loss = adversarial_loss(validity, valid)

        GenUpdater.zero_grad()
        g_loss.backward()
        GenUpdater.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Loss for real images
        real_pred, _, _ = discriminator(real_imgs)
        d_real_loss = adversarial_loss(real_pred, valid)

        # Loss for fake images
        fake_pred, _, _ = discriminator(gen_imgs.detach())
        d_fake_loss = adversarial_loss(fake_pred, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        DiscrimUpdater.zero_grad()
        d_loss.backward()
        DiscrimUpdater.step()

        # ------------------
        #  Information Loss
        # ------------------

        # Sample labels
        sampled_labels = np.random.randint(0, args.n_classes, batch_size)

        # Ground truth labels
        gt_labels = LongTensor(sampled_labels)

        # Sample noise, labels and code as generator input
        z = FloatTensor(np.random.normal(0, 1, (batch_size, args.latent_dim)))
        label_input = to_categorical(sampled_labels, num_columns=args.n_classes)
        code_input = FloatTensor(np.random.normal(-1, 1, (batch_size, args.code_dim)))

        gen_imgs = generator(z, label_input, code_input)
        _, pred_label, pred_code = discriminator(gen_imgs)

        info_loss = lambda_cat * categorical_loss(pred_label, gt_labels) + \
                    lambda_con * continuous_loss(pred_code, code_input)

        InfoUpdater.zero_grad()
        info_loss.backward()
        InfoUpdater.step()

        # --------------
        #  Log Progress
        # --------------

        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [info loss: %f]" % (
            epoch, args.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), info_loss.item()))

        batches_done = epoch * len(dataloader) + i
        if batches_done % args.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done)
