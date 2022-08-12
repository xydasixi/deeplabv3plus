import numpy as np
import os,random
from PIL import Image
import torch
from torchvision.transforms import functional as F
from torchvision import transforms as T

class data_import():
    def __init__(self, root_path, transforms=None):
        super(data_import, self).__init__()
        image_path = os.path.join(root_path, 'horse')
        mask_path = os.path.join(root_path, 'mask')
        image_list = os.listdir(image_path)
        mask_list = os.listdir(mask_path)
        self.img = [os.path.join(image_path, x ) for x in image_list]
        self.mask = [os.path.join(mask_path, x ) for x in mask_list]
        self.transforms = transforms
    def __getitem__(self, index):
        img = Image.open(self.img[index]).convert('RGB')
        target = Image.open(self.mask[index])
        img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.img)

class get_transform():
    def __init__(self, base_size, crop_size, train = True, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        trans = []
        if train:
            min_size = int(0.5 * base_size)
            max_size = int(2.0 * base_size)
            trans.extend([RandomResize(min_size, max_size), RandomHorizontalFlip(hflip_prob), RandomCrop(crop_size),])
        else:
            trans.append(RandomResize(base_size, base_size))
        trans.extend([ToTensor(),Normalize(mean=mean, std=std),])
        self.transforms = Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)

class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomResize():
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        # 这里size传入的是int类型，所以是将图像的最小边长缩放到size大小
        image = F.resize(image, size)
        # 这里的interpolation注意下，在torchvision(0.9.0)以后才有InterpolationMode.NEAREST
        # 如果是之前的版本需要使用PIL.Image.NEAREST
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target

class RandomHorizontalFlip():
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomCrop():
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = self.pad_if_smaller(image, self.size)
        target = self.pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target

    def pad_if_smaller(self, img, size, fill=0):
        # 如果图像最小边长小于给定size，则用数值fill进行padding
        min_size = min(img.size)
        if min_size < size:
            ow, oh = img.size
            padh = size - oh if oh < size else 0
            padw = size - ow if ow < size else 0
            img = F.pad(img, (0, 0, padw, padh), fill=fill)
        return img

class ToTensor():
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class Normalize():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

# if __name__ == '__main__':
#     root_path = os.path.join('data', 'weizmann_horse_db')
#     transforms = get_transform(base_size = 520, crop_size = 480, train = True)
#     l = data_import(root_path, transforms)[1]
#     a = 0