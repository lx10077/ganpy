import argparse
import time
import datetime
import sys
import os
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from models import *
from datasets import *
import torch
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="facades", help='name of the dataset')
parser.add_argument('--dataset_path', type=str, default="../data", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=256, help='size of image height')
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=500,
                    help='interval between sampling of images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=-1, help='interval between model checkpoints')
args = parser.parse_args()

os.makedirs('images/%s' % args.dataset_name, exist_ok=True)
os.makedirs('saved_models/%s' % args.dataset_name, exist_ok=True)
use_cuda = True if torch.cuda.is_available() else False

# Initialize generator and discriminator
generator = GeneratorUNet()
discriminator = Discriminator()

# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

if use_cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 100

# Calculate output of image discriminator (PatchGAN)
patch = (1, args.img_height // 2 ** 4, args.img_width // 2 ** 4)

if args.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load('saved_models/%s/generator_%d.pth' % (args.dataset_name, args.epoch)))
    discriminator.load_state_dict(torch.load('saved_models/%s/discriminator_%d.pth' % (args.dataset_name, args.epoch)))
else:
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

# Define optimizers for generator and discriminator
GenUpdater = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
DiscrimUpdater = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

# Configure dataloaders
transforms_ = [transforms.Resize((args.img_height, args.img_width), Image.BICUBIC),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
               ]

dataloader = DataLoader(
    ImageDataset(os.path.join(args.dataset_path, args.dataset_name), transforms_=transforms_),
    batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)

val_dataloader = DataLoader(
    ImageDataset(os.path.join(args.dataset_path, args.dataset_name), transforms_=transforms_, mode='val'),
    batch_size=10, shuffle=True, num_workers=1)

# Tensor type
Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    real_A = imgs['B'].type(Tensor)
    real_B = imgs['A'].type(Tensor)
    fake_B = generator(real_A)
    img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
    save_image(img_sample, 'images/%s/%s.png' % (args.dataset_name, batches_done), nrow=5, normalize=True)


prev_time = time.time()
for epoch in range(args.epoch, args.n_epochs):
    for i, batch in enumerate(dataloader):

        # Model inputs, the task is from B to A
        real_A = batch['B'].type(Tensor)
        real_B = batch['A'].type(Tensor)

        # Adversarial ground truths
        valid = Tensor(np.ones((real_A.size(0), *patch)))
        fake = Tensor(np.zeros((real_A.size(0), *patch)))

        # ------------------
        #  Train Generators
        # ------------------

        # GAN loss, G(A) should fake the discriminator
        fake_B = generator(real_A)
        pred_fake = discriminator(fake_B, real_A)
        loss_GAN = criterion_GAN(pred_fake, valid)

        # Pixel-wise loss, G(A) = B
        loss_pixel = criterion_pixelwise(fake_B, real_B)

        # Total loss, combime both losses
        loss_G = loss_GAN + lambda_pixel * loss_pixel

        GenUpdater.zero_grad()
        loss_G.backward()
        GenUpdater.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Real loss
        pred_real = discriminator(real_B, real_A)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss, stop backprop to the generator by detaching fake_B
        pred_fake = discriminator(fake_B.detach(), real_A)
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        DiscrimUpdater.zero_grad()
        loss_D.backward()
        DiscrimUpdater.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = args.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s" %
                         (epoch, args.n_epochs, i, len(dataloader), loss_D.item(), loss_G.item(),
                          loss_pixel.item(), loss_GAN.item(), time_left))

        # If at sample interval save image
        if batches_done % args.sample_interval == 0:
            sample_images(batches_done)

    if args.checkpoint_interval != -1 and epoch % args.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), 'saved_models/%s/generator_%d.pth' % (args.dataset_name, epoch))
        torch.save(discriminator.state_dict(), 'saved_models/%s/discriminator_%d.pth' % (args.dataset_name, epoch))
