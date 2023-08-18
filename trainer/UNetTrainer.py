#!/usr/bin/python3
"""
author:jiyaoliu

"""

import argparse
import itertools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from .utils import LambdaLR,Logger,ReplayBuffer,denorm
from .utils import weights_init_normal,get_config,plot_results,get_gaussian_kernel
from .datasets_unet import ImageDataset,ValDataset
from Model.UNet import *
from .utils import Resize,ToTensor,smooothing_loss
from .utils import Logger
from .reg import Reg
from torchvision.transforms import RandomAffine
from torchvision.transforms import RandomAffine,ToPILImage
from .transformer import Transformer_2D
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
import cv2
import os
from os.path import join
import lpips
from PIL import Image
import torch.nn.functional as F 
from trainer.utils import SoftDiceLoss
from trainer.utils import diceCoeffv2
from trainer.utils import EarlyStopping
import time
from tensorboardX import SummaryWriter
from tqdm import tqdm


class UNet_Trainer():
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.writer = SummaryWriter(os.path.join(config['save_root'], 'logtrain' ,str(int(time.time()))))
        self.val_writer = SummaryWriter(os.path.join(config['save_root'], 'logval' , str(int(time.time()))))

        ## def networks
        self.net= UNet(num_classes=config['num_classes'],depth=1).cuda()
       

        # 定义优化器
        if config['optimizer_type'] == 'adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config['initial_lr'], weight_decay=config['weight_decay'])
        else:
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.1, momentum=0.9)
                

        # # Inputs & targets memory allocation
        # Tensor = torch.cuda.FloatTensor if config['cuda'] else torch.Tensor
        # self.input_A = Tensor(config['batchSize'], config['input_nc'], config['size'], config['size'])
        # self.input_B = Tensor(config['batchSize'], config['num_classes'], config['size'], config['size'])


        # 定义早停机制
        self.early_stopping = EarlyStopping(config['early_stop_patience'], verbose=True, delta=config['lr_scheduler_eps'],
                                   path=os.path.join(config['save_root'], '{}.pth'.format(config['name'])))

        # Lossess
        self.loss = SoftDiceLoss(config['num_classes']).cuda()

        #Dataset loader
        self.dataloader = DataLoader(ImageDataset(config),
                                batch_size=config['batchSize'], shuffle=True, num_workers=config['n_cpu'])
        
        self.val_data = DataLoader(ValDataset(config),
                                batch_size=config['batchSize'], shuffle=False, num_workers=config['n_cpu'])
  


    def train(self):
        ###### Training ######
        for epoch in range(self.config['epoch'], self.config['n_epochs']):

            st = time.time()
            train_class_dices = np.array([0] * (self.config['num_classes'] - 1), dtype=np.float)
            val_class_dices = np.array([0] * (self.config['num_classes'] - 1), dtype=np.float)
            val_dice_arr = []
            train_losses = []
            val_losses = []

            # 训练模型
            self.net.train()
            for batch, pair in tqdm(enumerate(self.dataloader)):
                # print(pair['A'].shape)
                # X1 = Variable(self.input_A.copy_(pair['A']))
                # y = Variable(self.input_B.copy_(pair['B']))
                X1 = pair['A'].cuda()
                y = pair['B'].cuda()
                self.optimizer.zero_grad()
                output = self.net(X1)
                output = torch.sigmoid(output)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
                # iters += 1
                train_losses.append(loss.item())

                class_dice = []
                output = output.cpu().detach()
                y = y.cpu().detach()
                for i in range(1, self.config['num_classes']):
                    cur_dice = diceCoeffv2(output[:, i:i + 1, :], y[:, i:i + 1, :]).cpu().item()
                    class_dice.append(cur_dice)

                mean_dice = sum(class_dice) / len(class_dice)
                train_class_dices += np.array(class_dice)
                st = time.time()

            train_loss = np.average(train_losses)
            train_class_dices = train_class_dices / batch
            train_mean_dice = train_class_dices.sum() / train_class_dices.size

            self.writer.add_scalar('main_loss', train_loss, epoch)
            self.writer.add_scalar('main_dice', train_mean_dice, epoch)

            print('epoch {}/{} - train_loss: {:.4} - train_mean_dice: {:.4}'.format(
                    epoch, self.config['n_epochs'], train_loss, train_mean_dice))



                
            # 验证模型
            self.net.eval()
            for val_batch, pair in enumerate(self.val_data):
                # val_X1 = Variable(self.input_A.copy_(pair['A']))
                # val_y = Variable(self.input_A.copy_(pair['B']))
                val_X1 = pair['A'].cuda()
                val_y = pair['B'].cuda()

                pred = self.net(val_X1)
                pred = torch.sigmoid(pred)
                val_loss = self.loss(pred, val_y)
                val_losses.append(val_loss.item())
                pred = pred.cpu().detach()
                val_y = val_y.cpu().detach()
                val_class_dice = []
                for i in range(1, self.config['num_classes']):
                    val_class_dice.append(diceCoeffv2(pred[:, i:i + 1, :], val_y[:, i:i + 1, :]))

                val_dice_arr.append(val_class_dice)
                val_class_dices += np.array(val_class_dice)

            val_loss = np.average(val_losses)

            val_dice_arr = np.array(val_dice_arr)
            std = (np.std(val_dice_arr[:, 1:2]) + np.std(val_dice_arr[:, 2:3]) + np.std(val_dice_arr[:, 3:4])) / self.config['num_classes']
            val_class_dices = val_class_dices / val_batch

            val_mean_dice = val_class_dices.sum() / val_class_dices.size
            # organ_mean_dice = (val_class_dices[0] + val_class_dices[1] + val_class_dices[2]) / self.config['num_classes']

            self.val_writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], epoch)
            self.val_writer.add_scalar('main_loss', val_loss, epoch)
            self.val_writer.add_scalar('main_dice', val_mean_dice, epoch)
            # self.val_writer.add_scalar('lesion_dice', organ_mean_dice, epoch)

            print('val_loss: {:.4} - val_mean_dice: {:.4}'
                .format(val_loss, val_mean_dice, ))
            print('lr: {}'.format(self.optimizer.param_groups[0]['lr']))

            self.early_stopping(val_mean_dice, self.net, epoch)
            if self.early_stopping.early_stop or self.optimizer.param_groups[0]['lr'] < self.config['threshold_lr']:
                print("Early stopping")
                # 结束模型训练
                break

        print('----------------------------------------------------------')
        print('save epoch {}'.format(self.early_stopping.save_epoch))
        print('stoped epoch {}'.format(epoch))
        print('----------------------------------------------------------')           
                         
    

