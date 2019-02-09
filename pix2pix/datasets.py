import glob
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train'):
        dataset_name = os.path.basename(root)
        self.transform = transforms.Compose(transforms_)
        self.files = sorted(glob.glob(os.path.join(root, mode) + '/*.*'))

        # If there is no data locally, download it
        if len(self.files) == 0:
            print("Begin downloading the dataset: {} ...".format(dataset_name))
            save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
            site = "https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/{}.tar.gz".format(dataset_name)
            os.system("wget -P {} {}".format(save_dir, site))
            os.system("tar -zxvf {} -C {}".format(os.path.join(save_dir, "{}.tar.gz".format(dataset_name)), save_dir))
            self.files = sorted(glob.glob(os.path.join(root, mode) + '/*.*'))

        if mode == 'train':
            self.files.extend(sorted(glob.glob(os.path.join(root, 'test') + '/*.*')))

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w/2, h))
        img_B = img.crop((w/2, 0, w, h))

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], 'RGB')
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], 'RGB')

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {'A': img_A, 'B': img_B}

    def __len__(self):
        return len(self.files)
