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
from .utils import weights_init_normal,get_config,plot_results_v6,get_gaussian_kernel, plot_results_v6_test
from .datasets_v6 import ImageDataset,ValDataset,TestDataset
from Model.Pix2Pix_v6 import *
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


class P2p_Trainer_v6():
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.txt_path = config['train_txt_path']
        self.filenames = loadtxt(self.txt_path)
        ## def networks
        #original
        self.netG_A2B = Generator(config['input_nc'], 64,2,3,True,True).cuda()
        self.unet= UNet(num_classes=config['num_classes'],depth=1).cuda()
        self.unet.load_state_dict(torch.load(config['unet_chk']))


        # 定义四个判别器
        self.netD_B = [Discriminator(config['output_nc']*2).cuda(), 
                Discriminator(config['output_nc']*2).cuda(), 
                Discriminator(config['output_nc']*2).cuda(),
                Discriminator(config['output_nc']*2).cuda() ]
        #original
        self.optimizer_G = torch.optim.Adam(self.netG_A2B.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        self.optimizer_D_B = torch.optim.Adam(itertools.chain(self.netD_B[0].parameters(), self.netD_B[1].parameters(),
                                            self.netD_B[2].parameters(), self.netD_B[3].parameters()),
                                        lr=config['lr'], betas=(0.5, 0.999))
        if config['regist']:
            self.R_A=Reg(config['size'],config['size'],config['input_nc'],config['input_nc']).cuda()
            self.spatial_transform =Transformer_2D().cuda()
            self.optimizer_R_A=torch.optim.Adam(self.R_A.parameters(),lr=config['lr'],betas=(0.5,0.999))

        # Lossess
        self.MSE_loss = torch.nn.MSELoss()
        self.L1_loss = torch.nn.L1Loss()
        self.L12_loss=torch.nn.SmoothL1Loss(reduction='mean')
        self.perceptual_loss = lpips.LPIPS(net='vgg').cuda()
        self.softdice_loss = SoftDiceLoss(config['num_classes']).cuda() # seg loss

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if config['cuda'] else torch.Tensor
        self.input_A = Tensor(config['batchSize'], config['input_nc'], config['size'], config['size'])
        self.input_B = Tensor(config['batchSize'], config['output_nc'], config['size'], config['size'])
        self.input_C = Tensor(config['batchSize'], config['output_nc'], config['size'], config['size'])

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
                mask = Variable(self.input_C.copy_(batch['C']))

                self.optimizer_G.zero_grad()
                self.optimizer_R_A.zero_grad()
                fake_B = self.netG_A2B(real_A)

                #regist
                Trans =self.R_A(fake_B,real_B)
                SysRegist_A2B =self.spatial_transform(fake_B,Trans)  
                loss_SM=self.config['Corr_lamda']*self.L1_loss(SysRegist_A2B,real_B)  ###SR
                loss_SR=self.config['Smooth_lamda']* smooothing_loss(Trans)

                loss_L1 = self.L1_loss(SysRegist_A2B, real_B) * self.config['P2P_lamda']

                # gan loss: 遍历4张输入，分别和输出计算判别损失
                loss_GAN_A2B = 0
                for i in range(4):
                    fake_AB = torch.cat((real_A[:,i,:,:].unsqueeze(dim=0), SysRegist_A2B), 1) # 拼接两个[1,1,256,256]的矩阵为[1,2,256,256]
                    pred_fake = self.netD_B[i](fake_AB)
                    loss_GAN_A2B += self.MSE_loss(pred_fake, self.target_real) * self.config['Adv_lamda']


                # Gaussianblur loss:
                blur_layer = get_gaussian_kernel().cuda()
                SysRegist_A2B=blur_layer(SysRegist_A2B)  #用于计算L1_loss_blur
                fake_B1t=SysRegist_A2B.detach().cpu().numpy().squeeze() #用于计算train中的MAE/SSIM等指标 t作tensor
                real_B1=blur_layer(real_B)
                real_B1t=real_B1.detach().cpu().numpy().squeeze()
                loss_L1_blur=self.L1_loss(SysRegist_A2B,real_B1)*self.config['blur_lamda']
                


                
                # 只计算肝脏区域感知损失
                masked_real_B = torch.mul(real_B,mask)
                masked_SysRegist_B = torch.mul(SysRegist_A2B,mask)
                # perceptual_loss
                # loss_perceptual = self.perceptual_loss.forward(masked_fake_B, masked_real_B)*self.config['perceptual']  # v6
                loss_perceptual = self.perceptual_loss.forward(SysRegist_A2B, real_B)*self.config['perceptual'] # v6-1

                # shape 损失
                self.unet.eval()
                # 冻结参数
                for parm in self.unet.parameters():
                    parm.requires_grad = False
                # V6-6及之前
                # real_Bseg = torch.sigmoid(self.unet(real_B)) 
                # regist_Bseg = torch.sigmoid(self.unet(SysRegist_A2B))
                # loss_shape=F.mse_loss(regist_Bseg,real_Bseg)*self.config['shape']

                # V6-7
                real_Bseg = mask_to_onehot_v6(mask.cpu(), self.config['palette']).cuda() # 转为one-hot
                regist_Bseg = torch.sigmoid(self.unet(SysRegist_A2B)) 
                loss_shape = self.softdice_loss(regist_Bseg,real_Bseg) * self.config['shape']


                # Total loss 用gauss_blur后的图像计算
                toal_loss = loss_L1 + loss_GAN_A2B + loss_L1_blur + loss_perceptual + loss_SM + loss_SR + loss_shape #+ cross_loss+shape_loss
                toal_loss.backward()
                self.optimizer_G.step()
                self.optimizer_R_A.step()

                self.optimizer_D_B.zero_grad()
                with torch.no_grad():
                    fake_B = self.netG_A2B(real_A)

                loss_D_B = 0    
                for i in range(4):
                    pred_fake0 = self.netD_B[i](torch.cat((real_A[:,i,:,:].unsqueeze(dim=0), fake_B), 1)) 
                    pred_real = self.netD_B[i](torch.cat((real_A[:,i,:,:].unsqueeze(dim=0), real_B), 1)) 
                    loss_D_B += (self.MSE_loss(pred_fake0, self.target_fake)+self.MSE_loss(pred_real, self.target_real)) * self.config['Adv_lamda']

                
                loss_D_B.backward()
                self.optimizer_D_B.step()
                self.logger.log({'loss_D_B': loss_D_B ,'loss_G':toal_loss, 
                                 'loss_GAN_A2B':loss_GAN_A2B,
                                 'loss_L1':loss_L1, 
                                 'loss_L1_blur':loss_L1_blur, 
                                 'loss_perceptual':loss_perceptual,
                                 'loss_SM' : loss_SM,
                                 'loss_SR':loss_SR,
                                 'loss_shape':loss_shape
                },
                       images={'train_real_A': real_A, 'train_real_B': real_B, 'train_fake_B': fake_B, 'train_regist_B':SysRegist_A2B})#,'SR':SysRegist_A2B


                
    #         # Save models checkpoints
            if not os.path.exists(self.config["save_root"]):
                os.makedirs(self.config["save_root"])
            torch.save(self.netG_A2B.state_dict(), self.config['save_root'] + 'netG_A2B.pth')
            torch.save(self.R_A.state_dict(), self.config['save_root'] + 'Regist.pth')
            
            
            # #############val###############
            # with torch.no_grad():
            #     MAE, SSIM, PSNR,LPIPS = 0, 0, 0,0
            #     num = 0
            #     for i, batch in enumerate(self.val_data):
            #         real_A = Variable(self.input_A.copy_(batch['A']))
            #         real_B_tensor = Variable(self.input_B.copy_(batch['B']))
            #         real_B = real_B_tensor.detach().cpu().numpy().squeeze()
            #         fake_B_tensor = self.netG_A2B(real_A,mode='val')
            #         fake_B = fake_B_tensor.detach().cpu().numpy().squeeze()                   mae = self.MAE(fake_B,real_B)
            #         psnr = self.PSNR(fake_B,real_B)
            #         ssim = self.SSIM(fake_B,real_B)
            #         # Lpips =self.LPIPS(fake_B,real_B)
            #         MAE += mae
            #         PSNR += psnr
            #         SSIM += ssim 
            #         # LPIPS += Lpips
            #         num += 1

            #     # self.logger.log({'val_PSNR': PSNR/num,'val_SSIM':SSIM/num,'val_MAE':MAE/num,'val_LPIPS':LPIPS/num},
            #     #        images={'val_real_A': real_A, 'val_real_B': real_B_tensor, 'val_fake_B': fake_B_tensor,})#,'SR':SysRegist_A2B
            #     self.logger.log({'val_PSNR': PSNR/num,'val_SSIM':SSIM/num,'val_MAE':MAE/num},
            #            images={'val_real_A': real_A, 'val_real_B': real_B_tensor, 'val_fake_B': fake_B_tensor,})#,'SR':SysRegist_A2B

            #     # print ('MAE:',MAE/num)
                
                    
                         
    def validation(self,):
        self.netG_A2B.load_state_dict(torch.load(join(self.config['save_root'] + 'netG_A2B.pth')))
        self.R_A.load_state_dict(torch.load(join(self.config['save_root'] + 'Regist.pth')))
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
                    
                    real_B_tensor = Variable(self.input_B.copy_(batch['B']))
                    fake_B_tensor = self.netG_A2B(real_A)
                    Trans =self.R_A(fake_B_tensor,real_B_tensor)
                    # np.save('/home/zhouyc/jiyaoliu/Projects/Liver_PMX/Reg-GAN-PMX/script/def.npy',Trans.detach().cpu().numpy())
                    SysRegist_A2B =self.spatial_transform(fake_B_tensor,Trans)  
                    real_A_npy = real_A.detach().cpu().numpy().squeeze()
                    real_B = real_B_tensor.detach().cpu().numpy().squeeze()
                    fake_B = fake_B_tensor.detach().cpu().numpy().squeeze()
                    #regist

                    SysRegist_A2B_npy = SysRegist_A2B.detach().cpu().numpy().squeeze()

                    
                    # 绘制三幅测试结果
                    if num%20 == 0 and k<n:
                        plot_results_v6(self.config, num, real_A_npy, real_B, fake_B, SysRegist_A2B_npy)
                        self.save_deformation(Trans.squeeze(),join(self.config['image_save'],f'test_{num}_trans.png'))
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
                    # self.logger.log(
                    #    images={'val_real_A': real_A_tensor, 'val_real_B': real_B_tensor, 'val_fake_B': fake_B_tensor})
                    
                print ('MAE: {:.3f}+-{:.3f}'.format(np.mean(np.array(MAE)),np.std(np.array(MAE))))
                print ('PSNR: {:.3f}+-{:.3f}'.format(np.mean(np.array(PSNR)),np.std(np.array(PSNR))))
                print ('SSIM: {:.3f}+-{:.3f}'.format(np.mean(np.array(SSIM)),np.std(np.array(SSIM))))
                print('LPIPS: {:.3f}+-{:.3f}'.format(np.mean(np.array(LPIPS)),np.std(np.array(LPIPS))))


    # def test(self,):
    #     self.netG_A2B.load_state_dict(torch.load(join(self.config['save_root'] + 'netG_A2B.pth')))
    #     self.R_A.load_state_dict(torch.load(join(self.config['save_root'] + 'Regist.pth')))
    #     with torch.no_grad():
    #             MAE = 0
    #             PSNR = 0
    #             SSIM = 0
    #             LPIPS =0
    #             # num = 0
    #             from tqdm import tqdm
    #             k = 0
    #             n = 10 # 结果图的个数
    #             loss_fn=lpips.LPIPS(net='vgg').cuda()
    #             for num, batch in tqdm(enumerate(self.test_data)):
    #                 filename = self.filenames[num].split('.')[0]
    #                 real_A = Variable(self.input_A.copy_(batch['A']))
                    
    #                 real_B_tensor = Variable(self.input_B.copy_(batch['B']))
    #                 fake_B_tensor = self.netG_A2B(real_A)
    #                 Trans =self.R_A(fake_B_tensor,real_B_tensor)
    #                 SysRegist_A2B =self.spatial_transform(fake_B_tensor,Trans)  
    #                 real_A_npy = real_A.detach().cpu().numpy().squeeze()
    #                 real_B = real_B_tensor.detach().cpu().numpy().squeeze()
    #                 fake_B = fake_B_tensor.detach().cpu().numpy().squeeze()
    #                 #regist

    #                 SysRegist_A2B_npy = SysRegist_A2B.detach().cpu().numpy().squeeze()

                    
    #                 # 绘制测试结果
    #                 plot_results_v6_test(self.config, filename, real_A_npy, real_B, fake_B, SysRegist_A2B_npy)
    #                 self.save_deformation(Trans.squeeze(),join(self.config['save_root'] ,'img_test' ,filename ,f'{filename}_deformation.png'))


    #                 mae = self.MAE(fake_B,real_B)  
    #                 psnr = self.PSNR(fake_B,real_B)
    #                 ssim = self.SSIM(fake_B,real_B)
    #                 Lpips =self.LPIPS(loss_fn, fake_B_tensor,real_B_tensor)

    #                 MAE += mae
    #                 PSNR += psnr
    #                 SSIM += ssim 
    #                 LPIPS += Lpips
    #                 num += 1

    #             print ('MAE:',MAE/num)
    #             print ('PSNR:',PSNR/num)
    #             print ('SSIM:',SSIM/num)
    #             print('LPIPS:',LPIPS/num)

    def test(self,):
        self.netG_A2B.load_state_dict(torch.load(join(self.config['save_root'] + 'netG_A2B.pth')))
        self.R_A.load_state_dict(torch.load(join(self.config['save_root'] + 'Regist.pth')))
        with torch.no_grad():
                MAE = 0
                PSNR = 0
                SSIM = 0
                LPIPS =0
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
                    Trans =self.R_A(fake_B_tensor,real_B_tensor)
                    SysRegist_A2B =self.spatial_transform(fake_B_tensor,Trans)  
                    real_A_npy = real_A.detach().cpu().numpy().squeeze()
                    real_B = real_B_tensor.detach().cpu().numpy().squeeze()
                    fake_B = fake_B_tensor.detach().cpu().numpy().squeeze()
                    #regist

                    SysRegist_A2B_npy = SysRegist_A2B.detach().cpu().numpy().squeeze()

                    
                    # 绘制测试结果
                    
                    plot_results_v6_test(self.config, filename, real_A_npy, real_B, fake_B, SysRegist_A2B_npy)
                    self.save_deformation(Trans.squeeze(),join(self.config['save_root'] ,'img_test' ,filename ,f'{filename}_deformation.png'))


                    mae = self.MAE(fake_B,real_B)  
                    psnr = self.PSNR(fake_B,real_B)
                    ssim = self.SSIM(fake_B,real_B)
                    Lpips =self.LPIPS(loss_fn, fake_B_tensor,real_B_tensor)

                    MAE += mae
                    PSNR += psnr
                    SSIM += ssim 
                    LPIPS += Lpips
                    num += 1

                print ('MAE:',MAE/num)
                print ('PSNR:',PSNR/num)
                print ('SSIM:',SSIM/num)
                print('LPIPS:',LPIPS/num)

                  
            
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