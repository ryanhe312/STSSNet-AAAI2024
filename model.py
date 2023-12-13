import torch.nn.functional as F
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None ,kernel_size=3,padding=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class ConvUp(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=3,padding=1, bilinear=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2, kernel_size=kernel_size,
                               padding=padding)
    def forward(self, x1):
        x1 = self.up(x1)
        return self.conv(x1)
    
class LWGatedConv2D(nn.Module):
    def __init__(self, input_channel1, output_channel, pad, kernel_size, stride):
        super(LWGatedConv2D, self).__init__()

        self.conv_feature = nn.Conv2d(in_channels=input_channel1, out_channels=output_channel, kernel_size=kernel_size,
                                      stride=stride, padding=pad)

        self.conv_mask = nn.Sequential(
            nn.Conv2d(in_channels=input_channel1, out_channels=1, kernel_size=kernel_size, stride=stride,
                      padding=pad),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        newinputs = self.conv_feature(inputs)
        mask = self.conv_mask(inputs)

        return newinputs*mask
    
class DownLWGated(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = LWGatedConv2D(in_channels, in_channels, kernel_size=3, pad=1, stride=2)
        self.conv1 = LWGatedConv2D(in_channels, out_channels, kernel_size=3, stride=1, pad=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = LWGatedConv2D(out_channels, out_channels, kernel_size=3, stride=1, pad=1)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x= self.downsample(x)
        x= self.conv1(x)
        x = self.relu1(x)
        x= self.conv2(x)
        x = self.relu2(x)
        return x

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
      
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class STSSNet(nn.Module):
    def __init__(self, in_ch, out_ch, feat_ch, his_ch, skip=True):
        super(STSSNet, self).__init__()
        self.skip = skip

        self.convHis1 = nn.Sequential(
            nn.Conv2d(his_ch, 24, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.convHis2 = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.convHis3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.latentEncoder = nn.Sequential(
            nn.Conv2d(32+feat_ch, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size = 3, stride = 1, dilation = 1, padding = 1, bias=True)
        )
        self.KEncoder = nn.Sequential(
            nn.Conv2d(feat_ch, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size = 3, stride = 1, dilation = 1, padding = 1, bias=True)
        )

        self.lowlevelGated = LWGatedConv2D(32*3, 32, kernel_size=3, stride=1, pad=1)

        self.conv1 = LWGatedConv2D(in_ch+in_ch+feat_ch, 24, kernel_size=3, stride=1, pad=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = LWGatedConv2D(24, 24, kernel_size=3, stride=1, pad=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.down1 = DownLWGated(24, 24)
        self.down2 = DownLWGated(24, 32)
        self.down3 = DownLWGated(32, 32)
        
        self.up1 = Up(96+32, 32)
        self.up2 = Up(56, 24)
        self.up3 = Up(48, 24)
        self.outc = nn.Conv2d(24, out_ch*4, kernel_size=1)
        self.outfinal = nn.PixelShuffle(2)
        
    def hole_inpaint(self, x, mask, feature):
        x_down = x
        mask_down = F.interpolate(mask,scale_factor=0.125,mode='bilinear')
        feature_down = F.interpolate(feature,scale_factor=0.125,mode='bilinear')

        latent_code = self.latentEncoder(torch.cat([x_down,feature_down], dim=1)) * mask_down
        K_map = F.normalize(self.KEncoder(feature_down), p=2, dim=1)
        
        b,c,h,w = list(K_map.size())
        md = 2
        f1 = F.unfold(K_map*mask_down, kernel_size=(2*md+1, 2*md+1), padding=(md, md), stride=(1, 1))
        f1 = f1.view([b, c, -1, h, w])
        f2 = K_map.view([b, c, 1, h, w])
        weight_k = torch.relu((f1*f2).sum(dim=1, keepdim=True))
        
        b,c,h,w = list(latent_code.size())
        v = F.unfold(latent_code, kernel_size=(2*md+1, 2*md+1), padding=(md, md), stride=(1, 1))
        v = v.view([b, c, -1, h, w])
        
        agg_latent = (v * weight_k).sum(dim=2)/(weight_k.sum(dim=2).clamp_min(1e-6))
        return agg_latent
    
    def forward(self, x, feature, mask, hisBuffer):

        hisBuffer = hisBuffer.reshape(-1, 4, hisBuffer.shape[-2], hisBuffer.shape[-1])

        hisDown1 = self.convHis1(hisBuffer)
        hisDown2 = self.convHis2(hisDown1)
        hisDown3 = self.convHis3(hisDown2)
        cathisDown3 = hisDown3.reshape(-1, 3*32, hisDown3.shape[-2], hisDown3.shape[-1])  # 64

        motionFeature = self.lowlevelGated(cathisDown3)
        
        x1 = torch.cat([x, x*mask, feature],dim=1)
        x1 = self.conv1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.relu2(x1)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        inpaint_feat = self.hole_inpaint(x4, mask, feature)
        x4 = torch.cat([inpaint_feat, motionFeature], dim=1)

        res = self.up1(x4, x3)
        res= self.up2(res, x2)
        res= self.up3(res, x1)
        logits = self.outc(res)
        logits = self.outfinal(logits)

        if self.skip:
           x1, x2 = x.chunk(2,dim=1)
           x_up = F.interpolate(x1,scale_factor=2,mode='bilinear')
           logits = logits + x_up

        return logits