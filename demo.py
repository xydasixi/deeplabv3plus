from PIL import Image
import numpy as np
import torch
import os
import dataset
from torch.utils.data import DataLoader
from model.deeplab_model import DeeplabV3_plus
from dataset import data_load, get_transform

def mask_visual(img):
    """
    将mask文件转化为调色板模式，便于显示
    """
    mat = np.array(img)
    mat = mat.astype(np.uint8)

    # 实现array到image的转换
    dst = Image.fromarray(mat, 'P')

    bin_colormap = [211, 211, 211] + [70, 130, 180] + [255, 255, 255] * 253  # 二值调色板
    dst.putpalette(bin_colormap)
    return dst

def Pretreatment(root_path):
    """
    对预测的数据集进行预处理并保存到相关文件夹中
    """
    # 获取数据集的各个图片和mask的文件名
    image_list = os.listdir(os.path.join(root_path, 'horse'))
    mask_list = os.listdir(os.path.join(root_path, 'mask'))
    base_size = 460
    pad_size = 30

    # 定义预处理操作
    transforms = dataset.Compose([dataset.RandomResize(base_size, base_size),dataset.Pad(pad_size),dataset.CenterCrop(base_size + 2 * pad_size)])
    for i in range(len(image_list)):
        # 获取图片和mask文件路径并打开
        image_name =  os.path.join(os.path.join(root_path, 'horse'), image_list[i])
        mask_name = os.path.join(os.path.join(root_path, 'mask'), mask_list[i])
        image = Image.open(image_name).convert('RGB')
        mask = Image.open(mask_name)
        #对数据集进行预处理
        image_save,mask_save= transforms(image, mask)
        mask_save = mask_visual(mask_save)
        #保存预处理后的文件
        image_name = os.path.join(os.path.join(root_path, 'horse_pretreatment'), image_list[i])
        mask_name = os.path.join(os.path.join(root_path, 'mask_pretreatment'), mask_list[i])
        image_save.save(image_name)
        mask_save.save(mask_name)
    return mask_list

if __name__ == '__main__':
    root_path = 'image\\demo' #数据集根目录
    batchsize=5
    transforms = get_transform(base_size=460, pad_size=30, train=True)
    data = data_load(root_path=root_path, transforms=transforms)
    loader = DataLoader(data, batch_size=batchsize, num_workers=4)

    mask_list = Pretreatment(root_path)

    model = DeeplabV3_plus(num_classes=2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    weights_path = 'weights/best_weights'
    weights_file = os.listdir('weights/best_weights')[-1]
    weights = torch.load(os.path.join(weights_path, weights_file))
    model.load_state_dict(weights['model'])
    model.eval()

    for index, (image, target) in enumerate(loader):
        image, target = image.to(device), target.to(device)
        output = model(image).argmax(1)
        for i in range(batchsize):
            mask_pre = output[i]
            mask_pre = mask_visual(mask_pre.cpu().numpy())
            mask_name = os.path.join('image\\demo\\outputs', mask_list[index*batchsize+i])
            mask_pre.save(mask_name)


