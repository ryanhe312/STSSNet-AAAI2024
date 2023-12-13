# accelerated pytorch implementation for training

import torch
import numpy as np
import pytorch_msssim

class cvtColor:
    def __init__(self) -> None:
        scale1 = 1.0/255.0
        offset1 = 1.0/255.0
        self.rgb2ycbcr_coeffs = [
            65.481 * scale1,  128.553 * scale1, 24.966 * scale1, 16.0 * offset1, 
            -37.797 * scale1, -74.203 * scale1,  112.0 * scale1, 128.0 * offset1, 
            112.0 * scale1,  -93.786 * scale1, -18.214 * scale1, 128.0 * offset1]
        scale2 = 255.0
        offset2 = 1.0
        self.ycbcr2rgb_coeffs = [
            0.0045662 * scale2,  0,  0.0062589 * scale2, -0.8742024 * offset2,
            0.0045662 * scale2, -0.0015363 * scale2, -0.0031881 * scale2,  0.5316682 * offset2,
            0.0045662 * scale2,  0.0079107 * scale2,  0 ,  -1.0856326 * offset2
        ]
    def rgb2ycbcr(self, tensor):
        """
        tensor = B x C x H x W
        """
        R = tensor[:,0:1]
        G = tensor[:,1:2]
        B = tensor[:,2:3]

        Y  = self.rgb2ycbcr_coeffs[0] * R + self.rgb2ycbcr_coeffs[1] * G + self.rgb2ycbcr_coeffs[2]  * B + self.rgb2ycbcr_coeffs[3]
        Cb = self.rgb2ycbcr_coeffs[4] * R + self.rgb2ycbcr_coeffs[5] * G + self.rgb2ycbcr_coeffs[6]  * B + self.rgb2ycbcr_coeffs[7]
        Cr = self.rgb2ycbcr_coeffs[8] * R + self.rgb2ycbcr_coeffs[9] * G + self.rgb2ycbcr_coeffs[10] * B + self.rgb2ycbcr_coeffs[11]

        return torch.cat([Y,Cb,Cr],dim=1)

    def ycrcb2rgb(self, tensor):
        """
        tensor = B x C x H x W
        """

        Y = tensor[:,0:1]
        Cb = tensor[:,1:2]
        Cr = tensor[:,2:3]

        R = self.ycbcr2rgb_coeffs[0] * Y + self.ycbcr2rgb_coeffs[1] * Cb + self.ycbcr2rgb_coeffs[2]  * Cr + self.ycbcr2rgb_coeffs[3]
        G = self.ycbcr2rgb_coeffs[4] * Y + self.ycbcr2rgb_coeffs[5] * Cb + self.ycbcr2rgb_coeffs[6]  * Cr + self.ycbcr2rgb_coeffs[7]
        B = self.ycbcr2rgb_coeffs[8] * Y + self.ycbcr2rgb_coeffs[9] * Cb + self.ycbcr2rgb_coeffs[10] * Cr + self.ycbcr2rgb_coeffs[11]

        return torch.cat([R,G,B],dim=1)
    
cvtColor = cvtColor()

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        # print(target.shape)
        # print(pred.shape)
        # print(output.shape)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
        # print("c:", correct)
        # print('len(target):', len(target))
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def mse(output, target):
    with torch.no_grad():
        mse = (output - target).square().mean()
    return mse

def psnr(output, target, only_y=False):
    output = torch.clamp(output, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)
    if only_y:
        output = cvtColor.rgb2ycbcr(output)
        target = cvtColor.rgb2ycbcr(target)
        output = output[:,0:1]
        target = target[:,0:1]
    with torch.no_grad():
        mse = (output * 255.0 - target * 255.0).square().mean()
        psnr = 20.0 * torch.log10(255.0/mse.sqrt())
    return psnr

def ssim(output, target, only_y=False):
    output = torch.clamp(output, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)
    if only_y:
        output = cvtColor.rgb2ycbcr(output)
        target = cvtColor.rgb2ycbcr(target)
        output = output[:,0:1]
        target = target[:,0:1]
    # print(output.dtype,target.dtype)
    ssim = pytorch_msssim.ssim(output, target, data_range=1)
    return ssim