#!/usr/bin/python3
"""
author:Jiyao liu

"""

import argparse
import itertools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from .utils import LambdaLR,Logger,ReplayBuffer,denorm
from .utils import weights_init_normal,get_config,plot_results,get_gaussian_kernel, plot_results_v6_test
from .datasets_v6 import ImageDataset,ValDataset,TestDataset
from Model.CycleGan import *
from .utils import Resize,ToTensor,smooothing_loss, loadtxt, mkdir, mask_to_onehot_v6
from .utils import Logger
from .utils import SoftDiceLoss
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
from Model.UNet import *


class Cyc_Trainer():
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.txt_path = config['train_txt_path']
        self.filenames = loadtxt(self.txt_path)
        ## def networks
        self.netG_A2B = Generator(config['input_nc'], config['output_nc']).cuda()
        self.netD_B = Discriminator(config['output_nc']).cuda()
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        
        if config['bidirect']:
            self.netG_B2A = Generator(config['output_nc'], config['input_nc']).cuda()
            self.netD_A = Discriminator(config['input_nc']).cuda()
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),lr=config['lr'], betas=(0.5, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))

        if config['regist']:
            self.R_A=Reg(config['size'],config['size'],config['input_nc'],config['input_nc']).cuda()
            self.spatial_transform =Transformer_2D().cuda()
            self.optimizer_R_A=torch.optim.Adam(self.R_A.parameters(),lr=config['lr'],betas=(0.5,0.999))

        # Lossess
        self.MSE_loss = torch.nn.MSELoss()
        self.L1_loss = torch.nn.L1Loss()

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if config['cuda'] else torch.Tensor
        self.input_A = Tensor(config['batchSize'], config['input_nc'], config['size'], config['size'])
        self.input_B = Tensor(config['batchSize'], config['output_nc'], config['size'], config['size'])
        self.target_real = Variable(Tensor(1,1).fill_(1.0), requires_grad=False)
        self.target_fake = Variable(Tensor(1,1).fill_(0.0), requires_grad=False)

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

        #Dataset loader
        self.dataloader = DataLoader(ImageDataset(config),
                                batch_size=config['batchSize'], shuffle=True, num_workers=config['n_cpu'])
        
        self.val_data = DataLoader(ValDataset(config),
                                batch_size=config['batchSize'], shuffle=False, num_workers=config['n_cpu'])
        
        self.test_data = DataLoader(TestDataset(config),
                                batch_size=config['batchSize'], shuffle=False, num_workers=config['n_cpu'])

 
       # Loss plot
        self.logger = Logger(config['name'],config['port'],config['n_epochs'], len(self.dataloader))   


    def train(self):
        ###### Training ######
        for epoch in range(self.config['epoch'], self.config['n_epochs']):
            for i, batch in enumerate(self.dataloader):
                # Set model input
                real_A = Variable(self.input_A.copy_(batch['A']))
                real_B = Variable(self.input_B.copy_(batch['B']))
                if self.config['bidirect']:   # C dir= self.perceptual_loss.forward(fake_B, real_B)*self.config['perceptual'] # v6-1
                    self.optimizer_G.zero_grad()
                    # GAN loss
                    fake_B = self.netG_A2B(real_A)
                    pred_fake = self.netD_B(fake_B)
                    loss_GAN_A2B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)

                    fake_A = self.netG_B2A(real_B)
                    pred_fake = self.netD_A(fake_A)
                    loss_GAN_B2A = self.config['Adv_lamda']*self.MSE_loss(pred_fake, self.target_real)

                    # Cycle loss
                    recovered_A = self.netG_B2A(fake_B)
                    loss_cycle_ABA = self.config['Cyc_lamda'] * self.L1_loss(recovered_A, real_A)

                    recovered_B = self.netG_A2B(fake_A)
                    loss_cycle_BAB = self.config['Cyc_lamda'] * self.L1_loss(recovered_B, real_B)

                    # Total loss
                    loss_Total = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
                    loss_Total.backward()
                    self.optimizer_G.step()  

                    ###### Discriminator A ######
                    self.optimizer_D_A.zero_grad()
                    # Real loss
                    pred_real = self.netD_A(real_A)
                    loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)
                    # Fake loss
                    fake_A = self.fake_A_buffer.push_and_pop(fake_A)
                    pred_fake = self.netD_A(fake_A.detach())
                    loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)
 
                    # Total loss
                    loss_D_A = (loss_D_real + loss_D_fake)
                    loss_D_A.backward()

                    self.optimizer_D_A.step()
                    ###################################

                    ###### Discriminator B ######
                    self.optimizer_D_B.zero_grad()

                    # Real loss
                    pred_real = self.netD_B(real_B)
                    loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)

                    # Fake loss
                    fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                    pred_fake = self.netD_B(fake_B.detach())
                    loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)

                    # Total loss
                    loss_D_B = (loss_D_real + loss_D_fake)
                    loss_D_B.backward()

                    self.optimizer_D_B.step()
                    ###################################
                self.logger.log({'loss_D_B': loss_D_B, 'loss_D_A':loss_D_A,
                                 'loss_GAN_A2B' : loss_GAN_A2B, 'loss_GAN_B2A' : loss_GAN_B2A ,'loss_cycle_ABA' : loss_cycle_ABA ,'loss_cycle_BAB': loss_cycle_BAB},
                       images={'real_A': real_A, 'real_B': real_B, 'fake_B': fake_B, 'fake_A': fake_A})#,'SR':SysRegist_A2B






                
    #         # Save models checkpoints
            if not os.path.exists(self.config["save_root"]):
                os.makedirs(self.config["save_root"])
            torch.save(self.netG_A2B.state_dict(), self.config['save_root'] + 'netG_A2B.pth')      
                    
                         
    def validation(self,):
        self.netG_A2B.load_state_dict(torch.load(join(self.config['save_root'] + 'netG_A2B.pth')))
        # self.R_A.load_state_dict(torch.load(join(self.config['save_root'] + 'Regist.pth')))
        with torch.no_grad():
                MAE = []
                PSNR = []
                SSIM = []
                LPIPS =[]
                # num = 0
                from tqdm import tqdm
                k = 0
                n = 10 # 结果图的个数
                loss_fn=lpips.LPIPS(net='vgg').cuda()
                for num, batch in tqdm(enumerate(self.val_data)):
                    real_A = Variable(self.input_A.copy_(batch['A']))
                    real_A_npy = real_A.detach().cpu().numpy().squeeze()

                    real_B_tensor = Variable(self.input_B.copy_(batch['B']))
                    fake_B_tensor = self.netG_A2B(real_A)

                    real_B = real_B_tensor.detach().cpu().numpy().squeeze()
                    fake_B = fake_B_tensor.detach().cpu().numpy().squeeze()

                    
                    # 绘制三幅测试结果
                    if num%20 == 0 and k<n:
                        plot_results(self.config, num, real_A_npy, real_B, fake_B)
                        k +=1

                    mae = self.MAE(fake_B,real_B)  
                    psnr = self.PSNR(fake_B,real_B)
                    ssim = self.SSIM(fake_B,real_B)
                    Lpips =self.LPIPS(loss_fn, fake_B_tensor,real_B_tensor)
                    # print(mae,psnr,ssim,lpips)

                    MAE.append(mae)
                    PSNR.append(psnr)
                    SSIM.append(ssim)
                    LPIPS.append(Lpips.detach().cpu().numpy())
                    num += 1

                print ('MAE: {:.3f}+-{:.3f}'.format(np.mean(np.array(MAE)),np.std(np.array(MAE))))
                print ('PSNR: {:.3f}+-{:.3f}'.format(np.mean(np.array(PSNR)),np.std(np.array(PSNR))))
                print ('SSIM: {:.3f}+-{:.3f}'.format(np.mean(np.array(SSIM)),np.std(np.array(SSIM))))
                print('LPIPS: {:.3f}+-{:.3f}'.format(np.mean(np.array(LPIPS)),np.std(np.array(LPIPS))))

    def test(self,):
        self.netG_A2B.load_state_dict(torch.load(join(self.config['save_root'] + 'netG_A2B.pth')))
        with torch.no_grad():
                MAE = []
                PSNR = []
                SSIM = []
                LPIPS =[]
                # num = 0
                from tqdm import tqdm
                k = 0
                n = 10 # 结果图的个数
                test_filenames = loadtxt(self.config['test_txt_path'])
                loss_fn=lpips.LPIPS(net='vgg').cuda()
                for num, batch in tqdm(enumerate(self.test_data)):
                    filename = test_filenames[num].split('.')[0]
                    real_A = Variable(self.input_A.copy_(batch['A']))
                    
                    real_B_tensor = Variable(self.input_B.copy_(batch['B']))
                    fake_B_tensor = self.netG_A2B(real_A)

                    real_A_npy = real_A.detach().cpu().numpy().squeeze()
                    real_B = real_B_tensor.detach().cpu().numpy().squeeze()
                    fake_B = fake_B_tensor.detach().cpu().numpy().squeeze()


                    
                    # 绘制测试结果
                    
                    plot_results_v6_test(self.config, filename, real_A_npy, real_B, fake_B, fake_B)
                    # self.save_deformation(Trans.squeeze(),join(self.config['save_root'] ,'img_test' ,filename ,f'{filename}_deformation.png'))


                    mae = self.MAE(fake_B,real_B)  
                    psnr = self.PSNR(fake_B,real_B)
                    ssim = self.SSIM(fake_B,real_B)
                    Lpips =self.LPIPS(loss_fn, fake_B_tensor,real_B_tensor)

                    MAE.append(mae)
                    PSNR.append(psnr)
                    SSIM.append(ssim)
                    LPIPS.append(Lpips.detach().cpu().numpy())
                    num += 1

                print ('MAE: {:.3f}+-{:.3f}'.format(np.mean(np.array(MAE)),np.std(np.array(MAE))))
                print ('PSNR: {:.3f}+-{:.3f}'.format(np.mean(np.array(PSNR)),np.std(np.array(PSNR))))
                print ('SSIM: {:.3f}+-{:.3f}'.format(np.mean(np.array(SSIM)),np.std(np.array(SSIM))))
                print('LPIPS: {:.3f}+-{:.3f}'.format(np.mean(np.array(LPIPS)),np.std(np.array(LPIPS))))

            
            
    def MAE(self,fake,real):
        x,y = np.where(real!= -1)  # coordinate of target points
        #points = len(x)  #num of target points
        mae = np.abs(fake[x,y]-real[x,y]).mean()
            
        return mae/2    
            
            


    def PSNR(self,image,recon):

        image = self.normalize(image)
        recon = self.normalize(recon)

        return peak_signal_noise_ratio(image,recon, data_range=image.max())

    def SSIM(self,image,recon):

        image = self.normalize(image)
        recon = self.normalize(recon)

        return structural_similarity(image,recon, data_range=image.max())
    
    def LPIPS(self,loss_fn, fake,real):
        d=loss_fn.forward(fake,real)
        return d

    def normalize(self,image):
        """_summary_

        :param image: array in [0,1]
        :return: maxmum normalized image
        :rtype: array
        """
        MIN,MAX = np.min(image[:]),np.max(image[:])

        # 归一化到0-1
        im_normalized = image.copy()
        im_normalized = im_normalized - float(MIN)
        im_normalized = im_normalized * 1.0/(float(MAX)-float(MIN))

        return im_normalized

    def save_deformation(self,defms,root):
        heatmapshow = None
        defms_ = defms.data.cpu().float().numpy()
        dir_x = defms_[0]
        dir_y = defms_[1]
        x_max,x_min = dir_x.max(),dir_x.min()
        y_max,y_min = dir_y.max(),dir_y.min()
        dir_x = ((dir_x-x_min)/(x_max-x_min))*255
        dir_y = ((dir_y-y_min)/(y_max-y_min))*255
        tans_x = cv2.normalize(dir_x, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        tans_x[tans_x<=150] = 0
        tans_x = cv2.applyColorMap(tans_x, cv2.COLORMAP_JET)
        tans_y = cv2.normalize(dir_y, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        tans_y[tans_y<=150] = 0
        tans_y = cv2.applyColorMap(tans_y, cv2.COLORMAP_JET)
        gradxy = cv2.addWeighted(tans_x, 0.5,tans_y, 0.5, 0)

        cv2.imwrite(root, gradxy) 