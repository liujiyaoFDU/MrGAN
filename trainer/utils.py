import random
import cv2
import time
import datetime
import sys
import yaml
from torch.autograd import Variable
import torch
import torch.nn as nn
from visdom import Visdom
import torch.nn.functional as F
import numpy as np
import os
from os.path import splitext, isfile, join
import math
from PIL import Image
from torch.nn.modules.loss import _Loss
class Resize():
    def __init__(self, size_tuple, use_cv = True):
        self.size_tuple = size_tuple
        self.use_cv = use_cv


    def __call__(self, tensor):
        """
            Resized the tensor to the specific size

            Arg:    tensor  - The torch.Tensor obj whose rank is 4
            Ret:    Resized tensor
        """
        tensor = tensor.unsqueeze(0)
 
        tensor = F.interpolate(tensor, size = [self.size_tuple[0],self.size_tuple[1]])

        tensor = tensor.squeeze(0)
 
        return tensor#1, 64, 128, 128
class ToTensor():
    def __call__(self, tensor):
        tensor = np.expand_dims(tensor, 0)
        return torch.from_numpy(tensor)

def tensor2image(tensor):
    image = (127.5*(tensor.cpu().float().numpy()))+127.5
    image1 = image[0]
    for i in range(1,tensor.shape[0]):
        image1 = np.hstack((image1,image[i]))
    
    if image.shape[0] == 1:
        image = np.tile(image, (3, 1, 1))
    #print ('image1.shape:',image1.shape)
    return image1.astype(np.uint8)


class Logger():
    def __init__(self, env_name ,ports, n_epochs, batches_epoch):
        self.viz = Visdom(port= ports,env = env_name)
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}

    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write(
            '\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        if losses != None:
            for i, loss_name in enumerate(losses.keys()):
                if loss_name not in self.losses:
                    self.losses[loss_name] = losses[loss_name].item()
                else:
                    self.losses[loss_name] += losses[loss_name].item()

                if (i + 1) == len(losses.keys()):
                    sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name] / self.batch))
                else:
                    sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name] / self.batch))

            batches_done = self.batches_epoch * (self.epoch - 1) + self.batch
            batches_left = self.batches_epoch * (self.n_epochs - self.epoch) + self.batches_epoch - self.batch
            sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left * self.mean_period / batches_done)))

            # End of epoch
            if (self.batch % self.batches_epoch) == 0:
                # Plot losses
                for loss_name, loss in self.losses.items():
                    if loss_name not in self.loss_windows:
                        self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]),
                                                                    Y=np.array([loss / self.batch]),
                                                                    opts={'xlabel': 'epochs', 'ylabel': loss_name,
                                                                        'title': loss_name})
                    else:
                        self.viz.line(X=np.array([self.epoch]), Y=np.array([loss / self.batch]),
                                    win=self.loss_windows[loss_name], update='append')
                    # Reset losses for next epoch
                    self.losses[loss_name] = 0.0

                self.epoch += 1
                self.batch = 1
                sys.stdout.write('\n')
            else:
                self.batch += 1

        # Draw images
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data), opts={'title': image_name})
            else:
                self.viz.image(tensor2image(tensor.data), win=self.image_windows[image_name],
                               opts={'title': image_name})

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def weights_init_normal(m):
    # print ('m:',m)
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)
        
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

def smooothing_loss(y_pred):
    dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

    dx = dx*dx
    dy = dy*dy
    d = torch.mean(dx) + torch.mean(dy)
    grad = d 
    return d

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 调整对比度和亮度
def change_bri_crst(img,aa=0.75,bb=0): 
    # aa：对比度系数>1(增大对比度)<1（减小对比度）
    # bb ：亮度系数 >0(增大)<0（减小）
    img = ((img+ 1) / 2.0) * 255.0
    bri_mean = np.mean(img)
    idx = np.where(img<1)
    img_a = aa * (img-bri_mean) + bb + bri_mean
    img_a[idx]=0
    img_a = np.clip(img_a,0,255)
    return img_a

def plot_results_v6_test(config, filename, source_img ,target_img, rec_img, SysRegist_img):
    # save source
    save_img_path = join(config['save_root'] ,'img_test' ,filename)
    mkdir(save_img_path)

    source = (normalize(change_bri_crst(source_img))).astype(np.uint8)
    source_all = np.concatenate([source[i,:,:] for i in range(source.shape[0])],axis=1)
    image_PT = Image.fromarray(source_all)
    image_PT.save(join(save_img_path,f'{filename}_PT4.png'))
    # save target
    # np.save('/home/zhouyc/jiyaoliu/Projects/Liver_PMX/Reg-GAN-PMX/trainer/target_img.npy',target_img)
    target_img = Image.fromarray(normalize(change_bri_crst(target_img)).astype(np.uint8))
    rec_img = Image.fromarray(normalize(change_bri_crst(rec_img)).astype(np.uint8))
    SysRegist_img = Image.fromarray(normalize(change_bri_crst(SysRegist_img)).astype(np.uint8))
    target_img.save(join(save_img_path,f'{filename}_PMX_1.png'))
    rec_img.save(join(save_img_path,f'{filename}_PMX_2.png'))
    SysRegist_img.save(join(save_img_path,f'{filename}_PMX_3.png'))

def plot_results_v6(config, i, source_img ,target_img, rec_img, SysRegist_img):
    # save source
    mkdir(config['image_save'])
    source = (normalize((source_img+ 1) / 2.0) * 255.0).astype(np.uint8)
    source_all = np.concatenate([source[i,:,:] for i in range(source.shape[0])],axis=1)
    image_PT = Image.fromarray(source_all)
    image_PT.save(join(config['image_save'],f'test_{i}_PT.png'))
    # save target
    target_img = (normalize((target_img+ 1) / 2.0) * 255.0).astype(np.uint8)
    rec_img = (normalize((rec_img+ 1) / 2.0) * 255.0).astype(np.uint8)
    SysRegist_img = (normalize((SysRegist_img+ 1) / 2.0) * 255.0).astype(np.uint8)

    
    pair_image = np.concatenate([target_img,rec_img,SysRegist_img],1)
    image_PMX = Image.fromarray(pair_image)

    image_PMX.save(join(config['image_save'],f'test_{i}_PMX.png'))

def plot_results(config, i, source_img ,target_img, rec_img):
    # save source
    mkdir(config['image_save'])
    source = (normalize((source_img+ 1) / 2.0) * 255.0).astype(np.uint8)
    source_all = np.concatenate([source[i,:,:] for i in range(source.shape[0])],axis=1)
    image_PT = Image.fromarray(source_all)
    image_PT.save(join(config['image_save'],f'test_{i}_PT.png'))
    # save target
    target_img = (normalize((target_img+ 1) / 2.0) * 255.0).astype(np.uint8)
    rec_img = (normalize((rec_img+ 1) / 2.0) * 255.0).astype(np.uint8)
    pair_image = np.concatenate([target_img,rec_img],1)
    image_PMX = Image.fromarray(pair_image)

    image_PMX.save(join(config['image_save'],f'test_{i}_PMX.png'))


def loadtxt(txt_path):
    with open(txt_path,'r') as file:
        lines = file.readlines()
        lines = [line[:-1] for line in lines]  # 去掉换行符
        return lines
    
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def normalize(image):
    """_summary_

    :param image: array in [0,1]
    :return: maxmum normalized image
    :rtype: array
    """
    # MIN,MAX = np.min(image[:]),np.max(image[:])

    # # 归一化到0-1
    # im_normalized = image.copy()
    # im_normalized = im_normalized - float(MIN)
    # im_normalized = im_normalized * 1.0/(float(MAX)-float(MIN))

    im_normalized = image
    return im_normalized

def denorm(x):
    res = (x + 1.) / 2.
    res.clamp_(0, 1)
    return res

def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,kernel_size=kernel_size, groups=channels, bias=False, padding=kernel_size//2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    
    return gaussian_filter

# Unet utils
class SoftDiceLoss(_Loss):

    def __init__(self, num_classes):
        super(SoftDiceLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, y_pred, y_true):
        class_dice = []
        for i in range(1, self.num_classes):
            class_dice.append(diceCoeffv2(y_pred[:, i:i + 1, :], y_true[:, i:i + 1, :]))
        mean_dice = sum(class_dice) / len(class_dice)
        return 1 - mean_dice

def diceCoeffv2(pred, gt, eps=1e-5):
    r""" computational formula：
        dice = (2 * tp) / (2 * tp + fp + fn)
    """

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp
    score = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return score.sum() / N

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.save_epoch = 1

    def __call__(self, val_score, model, epoch):

        score = val_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model)
            self.save_epoch = epoch
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_score, model)
            self.save_epoch = epoch
            self.counter = 0

    def save_checkpoint(self, val_score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_score:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_score


def mask_to_onehot(mask, palette):
    """
    Converts a segmentation mask (C, H, W) to (C, H, W) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    mask = np.array(mask)
    semantic_map = []
    for colour in palette:
        equality = np.equal(mask, colour)
        # class_map = np.all(equality, axis=0)
        class_map = np.squeeze(equality)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=0).astype(np.float32)
    return torch.tensor(semantic_map)

def mask_to_onehot_v6(mask, palette):
    """
    Converts a segmentation mask (C, H, W) to (C, H, W) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    mask = np.array(mask)
    semantic_map = []
    for colour in palette:
        equality = np.equal(mask, colour)
        # class_map = np.all(equality, axis=0)
        class_map = np.squeeze(equality)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=1).astype(np.float32)
    return torch.tensor(semantic_map)
