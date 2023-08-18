"""
author:jiyao liu 20220223

Down: 1. 增加mask分支;
"""

import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import functools
from Model.CBAM import cbam_block
bias = False

class conv_block(nn.Module):
    # base block
    def __init__(self, ch_in, ch_out, affine=True, actv=nn.LeakyReLU(inplace=True), downsample=False, upsample=False):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.InstanceNorm2d(ch_out, affine=affine),
            actv,
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.InstanceNorm2d(ch_out, affine=affine),
            actv,
        )
        self.downsample = downsample
        self.upsample = upsample
        if self.upsample:
            self.up = up_conv(ch_out, ch_out // 2, affine)

    def forward(self, x):
        x1 = self.conv(x)
        c = x1.shape[1]
        if self.downsample:
            x2 = F.avg_pool2d(x1, 2)
            # half of channels for skip
            return x1[:,:c//2,:,:], x2
        # x1[:,:,:,:]
        if self.upsample:
            x2 = self.up(x1)
            return x2
        return x1


class up_conv(nn.Module):
    # base block
    def __init__(self, ch_in, ch_out, affine=True, actv=nn.LeakyReLU(inplace=True)):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.InstanceNorm2d(ch_out, affine=affine),
            actv,
        )

    def forward(self, x):
        x = self.up(x)
        return x

#Generator的encoder部分
class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, layers, affine):
        super(Encoder, self).__init__()
        encoder = []
        for i in range(layers):
            encoder.append(conv_block(input_nc, output_nc, affine, downsample=True, upsample=False))
            input_nc = output_nc
            output_nc = output_nc * 2
        self.encoder = nn.Sequential(*encoder)

    def forward(self, x):
        res = []
        for layer in self.encoder:
            x1, x2 = layer(x)
            res.append([x1, x2])
            x = x2
        return res

class ShareNet(nn.Module):
    # the Share Block of G
    def __init__(self, in_c, out_c, layers, affine,r):
        super(ShareNet, self).__init__()
        encoder = []
        decoder = []
        for i in range(layers-1):
            encoder.append(conv_block(in_c, in_c * 2, affine, downsample=True, upsample=False))
            decoder.append(conv_block(out_c-r, out_c//2, affine, downsample=False, upsample=True))
            in_c = in_c * 2
            out_c = out_c // 2
            r = r//2
        self.bottom = conv_block(in_c, in_c * 2, affine, upsample=True)
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)
        self.layers = layers

    def forward(self, x):
        encoder_output = []
        x = x[-1][1]
        for layer in self.encoder:
            x1,x2 = layer(x)
            encoder_output.append([x1, x2])
            x = x2
        bottom_output = self.bottom(x)
        if self.layers == 1:
            return bottom_output
        encoder_output.reverse()
        for i, layer in enumerate(self.decoder):
            x = torch.cat([bottom_output, encoder_output[i][0]], dim=1)
            x = layer(x)
            bottom_output = x
        return x

class Decoder(nn.Module):
    # the Decoder_x or Decoder_r of G
    def __init__(self, in_c, mid_c, layers, affine, r):
        super(Decoder, self).__init__()
        decoder = []
        for i in range(layers-1):
            decoder.append(
                nn.Sequential(cbam_block(in_c-r),
                            conv_block(in_c-r, mid_c, affine, downsample=False, upsample=True)))
            in_c = mid_c
            mid_c = mid_c // 2
            r = r//2
        self.conv_end = conv_block(in_c-r, mid_c, affine, downsample=False, upsample=False)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, share_input, encoder_input):
        encoder_input.reverse()
        x = 0
        ii = len(self.decoder)
        for i, (cbam,layer) in enumerate(self.decoder):
            x = torch.cat([share_input, encoder_input[i][0]], dim=1)
            # x = cbam(x)
            x = layer(x)
            share_input = x
        x = torch.cat([share_input, encoder_input[ii][0]], dim=1)
        x = self.conv_end(x)

        return x


class Generator(nn.Module):
    # the G of TarGAN

    def __init__(self, in_c, mid_c, layers, s_layers, affine, last_ac=True):
        super(Generator, self).__init__()
        self.img_encoder = Encoder(in_c, mid_c, layers, affine)
        self.img_decoder = Decoder(mid_c * (2 ** layers), mid_c * (2 ** (layers - 1)), layers, affine,64)
        self.share_net = ShareNet(mid_c * (2 ** (layers - 1)), mid_c * (2 ** (layers - 1 + s_layers)), s_layers, affine,256)
        self.out_img = nn.Conv2d(mid_c, 1, 1, bias=bias)
        self.last_ac = last_ac

    def forward(self, img):
        x_1 = self.img_encoder(img)
        s_1 = self.share_net(x_1)
        res_img = self.out_img(self.img_decoder(s_1,x_1))
        if self.last_ac:
            res_img = torch.tanh(res_img)
        return res_img


# class Discriminator(nn.Module):
#     # the D_x or D_r of TarGAN ( backbone of PatchGAN )

#     def __init__(self, image_size=256, conv_dim=64, c_dim=5, repeat_num=6):
#         super(Discriminator, self).__init__()
#         layers = []
#         layers.append(nn.Conv2d(1, conv_dim, kernel_size=4, stride=2, padding=1))
#         layers.append(nn.LeakyReLU(0.01))

#         curr_dim = conv_dim
#         for i in range(1, repeat_num):
#             layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
#             layers.append(nn.LeakyReLU(0.01))
#             curr_dim = curr_dim * 2

#         kernel_size = int(image_size / np.power(2, repeat_num))
#         self.main = nn.Sequential(*layers)
#         self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

#     def forward(self, x):
#         h = self.main(x)
#         out_src = self.conv1(h)  # [batch, 1, 4, 4] 
#         return out_src


class Discriminator(nn.Module):
    """Defines a PatchGAN discriminator,same as pix2pix"""

    def __init__(self, input_nc = 2, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(Discriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input) # [batch,1,30,30]


if __name__=='__main__':
    from torchsummary import summary
    import os
    # 生成器
    os.environ['CUDA_VISIBLE_DEVICES']='2'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = Generator(4, 64, 2, 3, True, True).to(device) # generator
    # model = Encoder(4, 64, 2,True).to(device)
    # summary(model, (4, 256, 256))
    # x = torch.tensor(np.zeros((1,4,256,256)),dtype=torch.float32).to(device)
    # print(model(x))
    # # 判别器
    # model_d = Discriminator().to(device)
    # import numpy as np
    # x = torch.tensor(np.zeros((1,1,256,256)),dtype=torch.float32).to(device)
    # summary(model_d,(1,256,256))
    # print(model_d(x))

    # sharenet
    # model = ShareNet(4, 64, 2, True, 256).to(device) # generator
    # model = Encoder(4, 64, 2,True).to(device)
    # summary(model, (64, 128, 128))

    # patchgan判别

    model =Discriminator(input_nc = 5, ndf=32, n_layers=2).to(device)
    summary(model, (5, 256, 256))

    # x = torch.zeros((1,4,256,256)).cuda()
    # generator = Generator(4, 64,2,3,True,True).cuda()
    # # decoder = Decoder(mid_c * (2 ** layers), mid_c * (2 ** (layers - 1)), layers, True,64)
    # print(generator(x,mode="val"))
    # # summary(decoder, ((128, 128, 128),(32, 256, 256)))