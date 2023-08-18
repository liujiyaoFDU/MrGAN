#!/usr/bin/python3
"""
author:jiyao liu 20230223


"""

import glob
import random
import os
import numpy as np
from os.path import splitext, isfile, join
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch


class ImageDataset(Dataset):
    def __init__(self, config):
        self.config= config
        self.txt_path = config['train_txt_path']
        self.filenames = loadtxt(self.txt_path)

        self.images_dir = config['images_dir']
        
    def __getitem__(self, index):

        name = self.filenames[index]
        images,mask = load_5_images_and_mask(self.images_dir, name)
        transform = get_transform(self.config,'train')
        transform_mask = get_transform(self.config,'mask_train')

        transformed_images = []
        # PT和 pmx/mask不同transform
        seed = np.random.randint(2147483647)
        for i in range(len(images[:-1])):
            # make a seed with numpy generator 
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            transformed_images.append(transform(images[i]))

        # pmx和max相同transform
        # seed = np.random.randint(2147483647) # 注释之后变为v6-2 、去掉注释为v6
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        transformed_images.append(transform(images[-1]))
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)  
        transformed_mask = transform_mask(mask)
        stack_input = torch.cat(transformed_images[:4],0)  
        
        return {'A': stack_input.float(), 'B': transformed_images[4].float(), 'C':transformed_mask.float()}

    def __len__(self):
        return len(self.filenames)


class ValDataset(Dataset):
    def __init__(self, config):
        self.config= config
        self.txt_path = config['val_txt_path']
        self.filenames = loadtxt(self.txt_path)

        self.images_dir = config['images_dir']
        
    def __getitem__(self, index):

        name = self.filenames[index]
        images,mask = load_5_images_and_mask(self.images_dir, name)
        transform = get_transform(self.config,'val')
        transform_mask = get_transform(self.config,'mask_val')

        transformed_images = []

        for i in range(len(images)):
            transformed_images.append(transform(images[i]))
            transformed_mask = transform_mask(mask)
        stack_input = torch.cat(transformed_images[:4],0)  
            
        return {'A': stack_input.float(), 'B': transformed_images[4].float(), 'C':transformed_mask.float()}
    def __len__(self):
        return len(self.filenames)
    
    
class TestDataset(Dataset):
    def __init__(self, config):
        self.config= config
        self.txt_path = config['test_txt_path']
        
        self.filenames = loadtxt(self.txt_path)

        self.images_dir = config['images_dir']
        
    def __getitem__(self, index):

        name = self.filenames[index]
        images,mask = load_5_images_and_mask(self.images_dir, name)
        transform = get_transform(self.config,'val')
        transform_mask = get_transform(self.config,'mask_val')

        transformed_images = []

        for i in range(len(images)):
            transformed_images.append(transform(images[i]))
            transformed_mask = transform_mask(mask)
        stack_input = torch.cat(transformed_images[:4],0)  
            
        return {'A': stack_input.float(), 'B': transformed_images[4].float(), 'C':transformed_mask.float()}
    def __len__(self):
        return len(self.filenames)


def loadtxt(txt_path):
    with open(txt_path,'r') as file:
        lines = file.readlines()
        lines = [line[:-1] for line in lines]  # 去掉换行符
        return lines


def load_5_images_and_mask(images_dir, filename):
    type_ = ['PT_T1_DYN_MSK', 'PT_T1_DYN_ART', 'PT_T1_DYN_POR', 'PT_T1_DYN_DEL','PMX_T1_DYN_HBP']
    images = []
    for ty in type_:
        images.append(preprocess(images_dir, ty, filename))
    mask = torch.tensor(np.load(join(images_dir, 'mask2', 'npy', filename))).unsqueeze(dim=0)
    return images, mask # shape 都为 [1, 174, 325]
    
def preprocess(images_dir, ty, filename):
    # 最大最小归一化0~1
    img_np = np.load(join(images_dir, ty, 'npy', filename))
    MIN,MAX = np.min(img_np[:]),np.max(img_np[:])

    im_array_sorted = img_np.copy()
    im_array_sorted = im_array_sorted - float(MIN)
    im_array_sorted = im_array_sorted * 1.0/float(MAX)
    
    return torch.tensor(im_array_sorted).unsqueeze(dim=0)


def get_transform(config, mode = 'train'):
    osize = config['resize']
    if mode == 'train':
        transform_list = [SquarePad(),
                            transforms.Resize(osize, Image.BICUBIC),
                            transforms.RandomAffine(degrees=1,translate=[0.02, 0.02],scale=[0.98, 1.02]),
                            transforms.Normalize((0.5,), (0.5,)),      
                    ]
    
    elif  mode == 'val':
        transform_list = [SquarePad(),
                            transforms.Resize(osize, Image.BICUBIC),
                            transforms.Normalize((0.5,), (0.5,)),      
                    ]    
    elif mode == 'mask_train':
        transform_list = [SquarePad(),
                            transforms.Resize(osize, Image.NEAREST),
                            transforms.RandomAffine(degrees=1,translate=[0.02, 0.02],scale=[0.98, 1.02])]
    elif mode == 'mask_val':
        transform_list = [SquarePad(),
                            transforms.Resize(osize, Image.NEAREST)]

    return transforms.Compose(transform_list)


class SquarePad:
    def __init__(self) -> None:
        pass
    def __call__(self, image):
        import copy
        img1 = copy.copy(image)
        img1 = img1.numpy()
        _,h,w = img1.shape
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)

        transform = transforms.Pad(padding, fill=0, padding_mode='constant')
        return transform(image)