import numpy as np
import os,random
from PIL import Image
import torch
from torchvision.transforms import functional as F
from torchvision import transforms as T

class data_load():
    """
    导入数据
    """
    def __init__(self, root_path, transforms=None):
        super(data_load, self).__init__()
        #获取数据集位置
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
        #对图片和mask进行预处理操作
        img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.img)

class get_transform():
    """
    定义图片和mask所需的预处理操作
    """
    def __init__(self, base_size, pad_size, train = True, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        trans = []
        # if train:
        #     min_size = int(0.5 * base_size)
        #     max_size = int(2.0 * base_size)
        #     trans.extend([RandomResize(min_size, max_size), RandomCrop(crop_size)])
        # else:

        # 将所需的预处理操作加到序列中
        trans.extend([RandomResize(base_size, base_size), Pad(pad_size), CenterCrop(base_size+2*pad_size)])
        trans.extend([ToTensor(),Normalize(mean=mean, std=std),])
        self.transforms = Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class Compose():
    """
    对图片和mask所需的多个预处理操作进行组合
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomResize():
    """
    缩放操作
    """
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        # 这里size传入的是int类型，所以是将图像的最小边长缩放到size大小
        image = F.resize(image, size)
        # mask resize操作为NEAREST
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class Pad():
    """
    填充操作
    """
    def __init__(self, pad_size):
        self.size = pad_size
    def __call__(self, image, target):
        size = self.size
        #对图片和mask四周填充，其中图片根据边缘像素值填充，mask填充0
        image = F.pad(image, size, padding_mode='edge')
        target = F.pad(target, size, fill=0)
        return image, target


class CenterCrop():
    """
    中心裁剪操作
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor():
    """
    转化为tensor
    """
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class Normalize():
    """
    图片标准化
    """
    def __init__(self, mean, std):
        # mean,std分别为均值和标准差
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

# class RandomHorizontalFlip():
#     def __init__(self, flip_prob):
#         self.flip_prob = flip_prob
#
#     def __call__(self, image, target):
#         if random.random() < self.flip_prob:
#             image = F.hflip(image)
#             target = F.hflip(target)
#         return image, target
#
#
# class RandomCrop():
#     def __init__(self, size):
#         self.size = size
#
#     def __call__(self, image, target):
#         image = self.pad_if_smaller(image, self.size)
#         target = self.pad_if_smaller(target, self.size)
#         crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
#         image = F.crop(image, *crop_params)
#         target = F.crop(target, *crop_params)
#         return image, target
#
#     def pad_if_smaller(self, img, size, fill=0):
#         # 如果图像最小边长小于给定size，则用数值fill进行padding
#         min_size = min(img.size)
#         if min_size < size:
#             ow, oh = img.size
#             padh = size - oh if oh < size else 0
#             padw = size - ow if ow < size else 0
#             img = F.pad(img, size, fill=fill)
#         return img

# if __name__ == '__main__':
#     root_path = os.path.join('data', 'weizmann_horse_db','train')
#     transforms = get_transform(base_size = 520, crop_size = 480, train = False)
#     l = data_load(root_path, transforms)[0]
#     l=0
