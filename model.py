import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class down(nn.Module):
    """
    A class for creating neural network blocks containing layers:
    
    Average Pooling --> Convlution + Leaky ReLU --> Convolution + Leaky ReLU
    
    This is used in the UNet Class to create a UNet like NN architecture.

    ...

    Methods
    -------
    forward(x)
        Returns output tensor after passing input `x` to the neural network
        block.
    """


    def __init__(self, inChannels, outChannels, filterSize):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the first convolutional layer.
            outChannels : int
                number of output channels for the first convolutional layer.
                This is also used as input and output channels for the
                second convolutional layer.
            filterSize : int
                filter size for the convolution filter. input N would create
                a N x N filter.
        """


        super(down, self).__init__()
        # Initialize convolutional layers.
        self.conv1 = nn.Conv2d(inChannels,  outChannels, filterSize, stride=1, padding=int((filterSize - 1) / 2))
        self.conv2 = nn.Conv2d(outChannels, outChannels, filterSize, stride=1, padding=int((filterSize - 1) / 2))
           
    def forward(self, x):
        """
        Returns output tensor after passing input `x` to the neural network
        block.

        Parameters
        ----------
            x : tensor
                input to the NN block.

        Returns
        -------
            tensor
                output of the NN block.
        """


        # Average pooling with kernel size 2 (2 x 2).
        x = F.avg_pool2d(x, 2)
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv1(x), negative_slope = 0.1)
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv2(x), negative_slope = 0.1)
        return x
    
class up(nn.Module):
    """
    A class for creating neural network blocks containing layers:
    
    Bilinear interpolation --> Convlution + Leaky ReLU --> Convolution + Leaky ReLU
    
    This is used in the UNet Class to create a UNet like NN architecture.

    ...

    Methods
    -------
    forward(x, skpCn)
        Returns output tensor after passing input `x` to the neural network
        block.
    """


    def __init__(self, inChannels, outChannels):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the first convolutional layer.
            outChannels : int
                number of output channels for the first convolutional layer.
                This is also used for setting input and output channels for
                the second convolutional layer.
        """

        
        super(up, self).__init__()
        # Initialize convolutional layers.
        self.conv1 = nn.Conv2d(inChannels,  outChannels, 3, stride=1, padding=1)
        # (2 * outChannels) is used for accommodating skip connection.
        self.conv2 = nn.Conv2d(2 * outChannels, outChannels, 3, stride=1, padding=1)
           
    def forward(self, x, skpCn):
        """
        Returns output tensor after passing input `x` to the neural network
        block.

        Parameters
        ----------
            x : tensor
                input to the NN block.
            skpCn : tensor
                skip connection input to the NN block.

        Returns
        -------
            tensor
                output of the NN block.
        """

        # Bilinear interpolation with scaling 2.
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv1(x), negative_slope = 0.1)
        # Convolution + Leaky ReLU on (`x`, `skpCn`)
        x = F.leaky_relu(self.conv2(torch.cat((x, skpCn), 1)), negative_slope = 0.1)
        return x


class UNet(nn.Module):
    """
    A class for creating UNet like architecture as specified by the
    Super SloMo paper.
    
    ...

    Methods
    -------
    forward(x)
        Returns output tensor after passing input `x` to the neural network
        block.
    """

    def __init__(self, inChannels, outChannels, device, decode_mode='deblur', seq_len=3):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the UNet.
            outChannels : int
                number of output channels for the UNet.
        """
        
        super(UNet, self).__init__()
        # Initialize neural network blocks.
        self.conv1 = nn.Conv2d(inChannels, 32, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.down1 = down(32, 64, 5)
        self.down2 = down(64, 128, 3)
        self.down3 = down(128, 256, 3)
        self.down4 = down(256, 512, 3)
        self.down5 = down(512, 512, 3)
        self.up1   = up(512, 512)
        self.up2   = up(512, 256)
        self.up3   = up(256, 128)
        self.up4   = up(128, 64)
        self.up5   = up(64, 32)
        self.conv3 = nn.Conv2d(32, outChannels, 3, stride=1, padding=1)

        self.up11 = up(512, 512)
        self.up12 = up(512, 256)
        self.up13 = up(256, 128)
        self.up14 = up(128, 64)
        self.up15 = up(64, 32)
        self.conv13 = nn.Conv2d(32, outChannels, 3, stride=1, padding=1)

        # self.self_attn = CALayer(512, 16)

        self.seq_len = seq_len
        self.decode_mode = decode_mode

        self.attn_mat = {}
        for f1 in range(int(self.seq_len)):
            for f2 in range(int(self.seq_len)):
                self.attn_mat['{}{}'.format(f1, f2)] = CrossCALayer(512, 16).to(device)
        self.attn_mat = nn.ModuleDict(self.attn_mat)

        self.inter_attn_mat = {}
        for f1 in range(int(self.seq_len)):
            for f2 in range(int(self.seq_len)):
                self.inter_attn_mat['{}{}'.format(f1, f2)] = CrossCALayer(512, 16).to(device)
        self.inter_attn_mat = nn.ModuleDict(self.inter_attn_mat)




    def encoder(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        s1 = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)
        pre_attn = self.down5(s5)

        pre_attn = pre_attn.unsqueeze(1)
        s1 = s1.unsqueeze(1)
        s2 = s2.unsqueeze(1)
        s3 = s3.unsqueeze(1)
        s4 = s4.unsqueeze(1)
        s5 = s5.unsqueeze(1)

        return pre_attn, s1, s2, s3, s4, s5

    def decoder(self, z, s1, s2, s3, s4, s5):
        z = self.up1(z, s5)
        z = self.up2(z, s4)
        z = self.up3(z, s3)
        z = self.up4(z, s2)
        z = self.up5(z, s1)
        z = F.leaky_relu(self.conv3(z), negative_slope=0.1)
        return z.unsqueeze(1)

    def decoder1(self, z, s1, s2, s3, s4, s5):
        z = self.up11(z, s5)
        z = self.up12(z, s4)
        z = self.up13(z, s3)
        z = self.up14(z, s2)
        z = self.up15(z, s1)
        z = F.leaky_relu(self.conv13(z), negative_slope=0.1)
        return z.unsqueeze(1)



    def forward(self, x):
        """
        Returns output tensor after passing input `x` to the neural network.

        Parameters
        ----------
            x : tensor
                input to the UNet.

        Returns
        -------
            tensor
                output of the UNet.
        """
        # import pdb; pdb.set_trace()
        for ii in range(x.size(1)):
            pre_attn_, s1_, s2_, s3_, s4_, s5_ = self.encoder(x[:, ii])

            if ii == 0:
                pre_attn = pre_attn_
                s1, s2, s3, s4, s5 = s1_, s2_, s3_, s4_, s5_
            else:
                pre_attn = torch.cat((pre_attn, pre_attn_), 1)
                s1 = torch.cat((s1, s1_), 1)
                s2 = torch.cat((s2, s2_), 1)
                s3 = torch.cat((s3, s3_), 1)
                s4 = torch.cat((s4, s4_), 1)
                s5 = torch.cat((s5, s5_), 1)
        # print('s5 shape', s5.shape)
        # print(pre_attn.shape, "encoder done")

        # for aa in range(pre_attn.size(1)):
        #     post_attn_ = self.attn_mat['{}{}'.format(aa, aa)](pre_attn[:, aa],pre_attn[:, aa]).unsqueeze(1)
        #
        #     for bb in range(aa + 1, aa + pre_attn.size(1)):
        #         post_attn_ += self.attn_mat['{}{}'.format(aa, bb % self.seq_len)]\
        #             (pre_attn[:, aa], pre_attn[:, bb % self.seq_len]).unsqueeze(1)
        #
        #     if aa == 0:
        #         post_attn = post_attn_
        #     else:
        #         post_attn = torch.cat((post_attn, post_attn_), 1)

        if self.decode_mode == 'interpolate':
            for aa in range(pre_attn.size(1)):
                post_attn_ = self.attn_mat['{}{}'.format(aa, aa)](pre_attn[:, aa], pre_attn[:, aa]).unsqueeze(1)

                for bb in range(aa + 1, aa + pre_attn.size(1)):
                    post_attn_ += self.attn_mat['{}{}'.format(aa, bb % self.seq_len)] \
                        (pre_attn[:, aa], pre_attn[:, bb % self.seq_len]).unsqueeze(1)

                if aa == 0:
                    post_attn = post_attn_
                else:
                    post_attn = torch.cat((post_attn, post_attn_), 1)

            for jj in range(post_attn.size(1)-1):
                # """
                x = self.decoder((post_attn[:, jj] + post_attn[:, jj+1])/2.0,
                                 (s1[:, jj] + s1[:, jj+ 1 ]) / 2.0,
                                 (s2[:, jj] + s2[:, jj + 1]) / 2.0,
                                 (s3[:, jj] + s3[:, jj + 1]) / 2.0,
                                 (s4[:, jj] + s4[:, jj + 1]) / 2.0,
                                 (s5[:, jj] + s5[:, jj + 1]) / 2.0
                                 )
                if jj == 0:
                    x_out = x
                else:
                    x_out = torch.cat((x_out, x), 1)

        elif self.decode_mode == 'deblur':
            for aa in range(pre_attn.size(1)):
                post_attn_ = self.attn_mat['{}{}'.format(aa, aa)](pre_attn[:, aa], pre_attn[:, aa]).unsqueeze(1)

                for bb in range(aa + 1, aa + pre_attn.size(1)):
                    post_attn_ += self.attn_mat['{}{}'.format(aa, bb % self.seq_len)] \
                        (pre_attn[:, aa], pre_attn[:, bb % self.seq_len]).unsqueeze(1)

                if aa == 0:
                    post_attn = post_attn_
                else:
                    post_attn = torch.cat((post_attn, post_attn_), 1)

            for jj in range(post_attn.size(1)):
                x = self.decoder(post_attn[:, jj], s1[:, jj], s2[:, jj], s3[:, jj], s4[:, jj], s5[:, jj])

            # print(x.shape)
                if jj == 0:
                    x_out = x
                else:
                    x_out = torch.cat((x_out, x), 1)
            # input(x_out.shape)
        else:
            # input('BOTH')
            for aa in range(pre_attn.size(1)):
                post_attn_ = self.attn_mat['{}{}'.format(aa, aa)](pre_attn[:, aa], pre_attn[:, aa]).unsqueeze(1)
                inter_post_attn_ = self.inter_attn_mat['{}{}'.format(aa, aa)](pre_attn[:, aa],
                                                                              pre_attn[:, aa]).unsqueeze(1)

                for bb in range(aa + 1, aa + pre_attn.size(1)):
                    post_attn_ += self.attn_mat['{}{}'.format(aa, bb % self.seq_len)] \
                        (pre_attn[:, aa], pre_attn[:, bb % self.seq_len]).unsqueeze(1)
                    inter_post_attn_ += self.inter_attn_mat['{}{}'.format(aa, bb % self.seq_len)] \
                        (pre_attn[:, aa], pre_attn[:, bb % self.seq_len]).unsqueeze(1)

                if aa == 0:
                    post_attn = post_attn_
                    inter_post_attn = inter_post_attn_
                else:
                    post_attn = torch.cat((post_attn, post_attn_), 1)
                    inter_post_attn = torch.cat((inter_post_attn, inter_post_attn_), 1)

            for jj in range(post_attn.size(1)):
                x = self.decoder(post_attn[:, jj], s1[:, jj], s2[:, jj], s3[:, jj], s4[:, jj], s5[:, jj])

            # print(x.shape)
                if jj == 0:
                    x_out = x
                else:
                    x_out = torch.cat((x_out, x), 1)

                if jj < post_attn.size(1) - 1:
                    x = self.decoder1((inter_post_attn[:, jj] + inter_post_attn[:, jj + 1]) / 2.0,
                                      (s1[:, jj] + s1[:, jj + 1]) / 2.0,
                                      (s2[:, jj] + s2[:, jj + 1]) / 2.0,
                                      (s3[:, jj] + s3[:, jj + 1]) / 2.0,
                                      (s4[:, jj] + s4[:, jj + 1]) / 2.0,
                                      (s5[:, jj] + s5[:, jj + 1]) / 2.0
                                      )
                    x_out = torch.cat((x_out, x), 1)

        return x_out


class backWarp(nn.Module):
    """
    A class for creating a backwarping object.

    This is used for backwarping to an image:

    Given optical flow from frame I0 to I1 --> F_0_1 and frame I1, 
    it generates I0 <-- backwarp(F_0_1, I1).

    ...

    Methods
    -------
    forward(x)
        Returns output tensor after passing input `img` and `flow` to the backwarping
        block.
    """


    def __init__(self, W, H, device):
        """
        Parameters
        ----------
            W : int
                width of the image.
            H : int
                height of the image.
            device : device
                computation device (cpu/cuda). 
        """


        super(backWarp, self).__init__()
        # create a grid
        gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
        self.W = W
        self.H = H
        self.gridX = torch.tensor(gridX, requires_grad=False, device=device)
        self.gridY = torch.tensor(gridY, requires_grad=False, device=device)
        
    def forward(self, img, flow):
        """
        Returns output tensor after passing input `img` and `flow` to the backwarping
        block.
        I0  = backwarp(I1, F_0_1)

        Parameters
        ----------
            img : tensor
                frame I1.
            flow : tensor
                optical flow from I0 and I1: F_0_1.

        Returns
        -------
            tensor
                frame I0.
        """


        # Extract horizontal and vertical flows.
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        x = self.gridX.unsqueeze(0).expand_as(u).float() + u
        y = self.gridY.unsqueeze(0).expand_as(v).float() + v
        # range -1 to 1
        x = 2*(x/self.W - 0.5)
        y = 2*(y/self.H - 0.5)
        # stacking X and Y
        grid = torch.stack((x,y), dim=3)
        # Sample pixels using bilinear interpolation.
        imgOut = torch.nn.functional.grid_sample(img, grid)
        return imgOut


# Creating an array of `t` values for the 7 intermediate frames between
# reference frames I0 and I1. 
t = np.linspace(0.125, 0.875, 7)

def getFlowCoeff (indices, device):
    """
    Gets flow coefficients used for calculating intermediate optical
    flows from optical flows between I0 and I1: F_0_1 and F_1_0.

    F_t_0 = C00 x F_0_1 + C01 x F_1_0
    F_t_1 = C10 x F_0_1 + C11 x F_1_0

    where,
    C00 = -(1 - t) x t
    C01 = t x t
    C10 = (1 - t) x (1 - t)
    C11 = -t x (1 - t)

    Parameters
    ----------
        indices : tensor
            indices corresponding to the intermediate frame positions
            of all samples in the batch.
        device : device
                computation device (cpu/cuda). 

    Returns
    -------
        tensor
            coefficients C00, C01, C10, C11.
    """


    # Convert indices tensor to numpy array
    ind = indices.detach().numpy()
    C11 = C00 = - (1 - (t[ind])) * (t[ind])
    C01 = (t[ind]) * (t[ind])
    C10 = (1 - (t[ind])) * (1 - (t[ind]))
    return torch.Tensor(C00)[None, None, None, :].permute(3, 0, 1, 2).to(device), torch.Tensor(C01)[None, None, None, :].permute(3, 0, 1, 2).to(device), torch.Tensor(C10)[None, None, None, :].permute(3, 0, 1, 2).to(device), torch.Tensor(C11)[None, None, None, :].permute(3, 0, 1, 2).to(device)

def getWarpCoeff (indices, device):
    """
    Gets coefficients used for calculating final intermediate 
    frame `It_gen` from backwarped images using flows F_t_0 and F_t_1.

    It_gen = (C0 x V_t_0 x g_I_0_F_t_0 + C1 x V_t_1 x g_I_1_F_t_1) / (C0 x V_t_0 + C1 x V_t_1)

    where,
    C0 = 1 - t
    C1 = t

    V_t_0, V_t_1 --> visibility maps
    g_I_0_F_t_0, g_I_1_F_t_1 --> backwarped intermediate frames

    Parameters
    ----------
        indices : tensor
            indices corresponding to the intermediate frame positions
            of all samples in the batch.
        device : device
                computation device (cpu/cuda). 

    Returns
    -------
        tensor
            coefficients C0 and C1.
    """


    # Convert indices tensor to numpy array
    ind = indices.detach().numpy()
    C0 = 1 - t[ind]
    C1 = t[ind]
    return torch.Tensor(C0)[None, None, None, :].permute(3, 0, 1, 2).to(device), torch.Tensor(C1)[None, None, None, :].permute(3, 0, 1, 2).to(device)


## Channel Attention (CA) Layer
class CrossCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CrossCALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x1, x2):
        #input("Input shape: {}".format(x.shape))
        y = self.avg_pool(x1)
        #input("Avg pool shape: {}".format(y.shape))
        y = self.conv_du(y)
        with open('./attention.txt','a+') as f:
            
            f.write(str(y.detach().cpu().numpy()[0]))
        #input("Attn shape: {}".format(y.shape))
        return x2 * y

## Channel Attention (CA) Layer
class InterLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(InterLayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du1 = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
        self.conv_du2 = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x1, x2):
        #input("Input shape: {}".format(x.shape))
        w1 = self.conv_du1(self.avg_pool1(x1))        
        w2 = self.conv_du2(self.avg_pool2(x2))  
        y = (w1*x1 + w2*x2)/(w1+w2)

        return y
