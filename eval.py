import os
import cv2
import lpips
import torch
import numpy as np
import torch.nn.functional as F
import torch.utils.data as data

from tqdm import tqdm

from utils import matlab_metric, metrics
from dataloaders import *

def ImgWrite(mPath,prefix,idx,img):
    cv2.imwrite(os.path.join(mPath,prefix+"."+str(idx).zfill(4)+".png"),img)

@torch.no_grad()
def save_res(dataLoaderIns, model, modelPath, save_dir, save_img=True, mode='all'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if modelPath.endswith(".tar"):
        model_CKPT = torch.load(modelPath, map_location="cuda:0")["state_dict"]
    elif modelPath.endswith(".ckpt"):
        model_CKPT = {k[6:]:v for k,v in torch.load(modelPath, map_location="cuda:0")["state_dict"].items() if 'vgg' not in k}
    else:
        model_CKPT = torch.load(modelPath, map_location="cuda:0")
    model.load_state_dict(model_CKPT)
    model = model.to("cuda:0")
    model.eval()

    all_PSNR_SF = []
    all_ssim_SF = []
    all_lpips_SF = []
    
    all_PSNR_IF = []
    all_ssim_IF = []
    all_lpips_IF = []
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()


    print('saving to ',save_dir)
    f = open(os.path.join(save_dir, 'metrics.csv'), 'w')
    print('frame,psnr,ssim,lpips', file=f)
    for index, (input,features,mask,hisBuffer,label) in tqdm(dataLoaderIns):
        index = index[0].item()
        input=input.cuda()
        hisBuffer=hisBuffer.cuda()
        mask=mask.cuda()
        features=features.cuda()
        label=label.cuda()
        
        B,C,H,W = input.size()

        input = F.pad(input,(0,0,0,4),'replicate')
        mask = F.pad(mask,(0,0,0,4),'replicate')
        features = F.pad(features,(0,0,0,4),'replicate')
        hisBuffer = F.pad(hisBuffer.reshape(B,-1,H,W),(0,0,0,4),'replicate').reshape(B,3,4,H+4,W)
        
        res=model(input, features, mask, hisBuffer)
        res = res[:,:,:-8]

        ## mask
        if mode == 'edge':
            gray = cv2.cvtColor((label[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            mask = cv2.Canny(gray, 100, 200)
        elif mode == 'hole':
            mask = 1 - mask[:, :, :-4]
            mask = F.interpolate(mask, scale_factor=2, mode='bilinear').squeeze().cpu().numpy()
        else:
            mask = None

        ## calculate metrics
        psnr, ssim = matlab_metric.calc_metrics(res[0].permute(1,2,0).detach().cpu().numpy(), label[0].permute(1,2,0).detach().cpu().numpy(), 0, norm=True, mask=mask)
        with torch.no_grad():
            lpips_ = loss_fn_alex(res, label).item()
        
        if index % 2 == 0:
            all_PSNR_SF.append(psnr)
            all_ssim_SF.append(ssim)
            all_lpips_SF.append(lpips_)
        else:
            all_PSNR_IF.append(psnr)
            all_ssim_IF.append(ssim)
            all_lpips_IF.append(lpips_)

        print(index, psnr, ssim, lpips_, file=f, sep=',', flush=True)

        ## save res
        if save_img:
            res=res.squeeze(0).cpu().numpy().transpose([1,2,0])
            res=cv2.cvtColor(res,cv2.COLOR_RGB2BGR)
            res = (np.clip(res, 0, 1) * 255).astype(np.uint8)
            ImgWrite(save_dir,"res",index,res)

    psnr_sf = np.mean(all_PSNR_SF)
    ssim_sf = np.mean(all_ssim_SF)
    lpips_sf = np.mean(all_lpips_SF)
    
    psnr_if = np.mean(all_PSNR_IF)
    ssim_if = np.mean(all_ssim_IF)
    lpips_if = np.mean(all_lpips_IF)
    
    print('SF', psnr_sf, ssim_sf, lpips_sf, file=f, sep=',')
    print('IF', psnr_if, ssim_if, lpips_if, file=f, sep=',')
    print('MEAN', (psnr_sf+psnr_if)/2, (ssim_sf+ssim_if)/2, (lpips_if+lpips_sf)/2, file=f, sep=',')

    f.close()

def plot_res(save_dir):
    import matplotlib.pyplot as plt
    with open(os.path.join(save_dir, 'metrics.csv'), 'r') as f:
        data = np.loadtxt(f, delimiter=',', skiprows=1, usecols=(1,2,3))[:-3]
        all_PSNR_SF, all_ssim_SF, all_lpips_SF = data[::2,0], data[::2,1], data[::2,2]
        all_PSNR_IF, all_ssim_IF, all_lpips_IF = data[1::2,0], data[1::2,1], data[1::2,2]
    
    plt.plot(np.arange(len(all_PSNR_SF)), np.array(all_PSNR_SF))
    plt.plot(np.arange(len(all_PSNR_IF)), np.array(all_PSNR_IF))
    plt.legend(['PSNR-SF', 'PSNR-IF'])
    plt.savefig(os.path.join(save_dir, 'PSNR-curve.jpg'))
 
if __name__ =="__main__":
    # Lewis
    dataset = get_Lewis_test_data()
    modelPath = 'checkpoints/Lewis/finalModel.pkl'
    savePath = 'output/Lewis'

    # Subway
    # dataset = get_Subway_test_data()
    # modelPath = 'checkpoints/Subway/finalModel.pkl'
    # savePath = 'output/Subway'

    # SunTemple
    # dataset = get_SunTemple_test_data()
    # modelPath = 'checkpoints/SunTemple/finalModel.pkl'
    # savePath = 'output/SunTemple'
    
    # Arena
    # dataset = get_Arena_test_data()
    # modelPath = 'checkpoints/Arena/finalModel.pkl'
    # savePath = 'output/Arena'

    testLoader = data.DataLoader(dataset,1,shuffle=False,num_workers=2, pin_memory=True)
    
    from model import STSSNet
    model = STSSNet(6,3,9,4)

    mode = 'all' # 'all', 'edge', 'hole'

    save_res(testLoader, model, modelPath, savePath, save_img=False, mode=mode)
    plot_res(savePath)
   
