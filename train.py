import os
import time
import torch
import lpips
import torchvision as tv
import torch.nn.functional as F
import torch.utils.data as data

from torch import optim
from torch.cuda import amp
from visdom import Visdom
from model import STSSNet
from tqdm.auto import tqdm

from dataloaders import *
from utils import metrics

mdevice=torch.device("cuda:0")
learningrate=1e-4
epoch=100
printevery=50
batch_size=2

class VisdomWriter:
    def __init__(self, visdom_port):
        self.viz = Visdom(port=visdom_port)
        self.names = []
    def add_scalar(self, name, val, step):
        try:
            val = val.item()
        except:
            val = float(val)
        if name not in self.names:
            self.names.append(name)
            self.viz.line([val], [step], win=name, opts=dict(title=name))
        else:
            self.viz.line([val], [step], win=name, update='append')
    def add_image(self, name, image, step):
        self.viz.image(image, win=name, opts=dict(title=name))
    def close(self):
        return

def colornorm(img):
    img = img.clamp(0,1)
    return img

def train(dataLoaderIns, modelSavePath, save_dir, reload=None, port=2336):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    vgg_model = lpips.LPIPS(net='vgg').cuda()

    model = STSSNet(6,3,9,4)

    model = model.to(mdevice)
    scaler = amp.GradScaler()
    optimizer = optim.Adam(model.parameters(), lr=learningrate)

    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.9)

    writer = VisdomWriter(port)
    global_step = 0
    start_e = 0

    if reload is not None:
        pth = torch.load(os.path.join(save_dir, f'totalModel.{reload}.pth.tar'))
        start_e = pth['epoch']
        model.load_state_dict(pth['state_dict'])
        optimizer.load_state_dict(pth['optimizer'])
        for e in range(start_e):
            if e > 20:
                scheduler.step()

    print('start epoch:', start_e)
    for e in range(start_e, epoch):
        model.train()

        iter=0
        loss_all=0
        startTime = time.time()

        for input,features,mask,hisBuffer,label in tqdm(dataLoaderIns):

            input=input.cuda()
            hisBuffer=hisBuffer.cuda()
            mask=mask.cuda()
            features=features.cuda()
            label=label.cuda()

            input_lst, hisBuffer_lst, mask_lst, features_lst, label_lst = [], [], [], [], []
            for i in range(4):
                i, j, h, w = tv.transforms.RandomCrop.get_params(input, output_size=(256, 256))
                input_lst.append(tv.transforms.functional.crop(input, i, j, h, w))
                hisBuffer_lst.append(tv.transforms.functional.crop(hisBuffer, i, j, h, w))
                mask_lst.append(tv.transforms.functional.crop(mask, i, j, h, w))
                features_lst.append(tv.transforms.functional.crop(features, i, j, h, w))
                label_lst.append(tv.transforms.functional.crop(label, i*2, j*2, h*2, w*2))
            input, hisBuffer, mask, features, label = torch.cat(input_lst),torch.cat(hisBuffer_lst), torch.cat(mask_lst), torch.cat(features_lst), torch.cat(label_lst)

            optimizer.zero_grad()

            with amp.autocast():
                res=model(input, features, mask, hisBuffer).float()
                loss_full = torch.abs(res-label).mean()
                mask_up = (F.interpolate(mask,scale_factor=2,mode='bilinear') > 0).float()
                loss_mask = (torch.abs(res-label)*(1-mask_up)).sum()/(1-mask_up).sum().clamp_min(1e-6)
                loss_lpips = vgg_model(res*2 -1,label*2-1)
                loss = loss_full + loss_mask +  0.01 * loss_lpips

            scaler.scale(loss.mean()).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if iter % printevery == 0:
                with torch.no_grad():
                    writer.add_scalar('loss/loss_total', loss.mean(), global_step)
                    writer.add_scalar('loss/loss_full', loss_full.mean(), global_step)
                    writer.add_scalar('loss/loss_mask', loss_mask.mean(), global_step)
                    writer.add_scalar('loss/loss_lpips', loss_lpips.mean(), global_step)

                    writer.add_image('img/input', colornorm(input[-1,:3]).cpu().detach(), global_step)
                    writer.add_image('img/gt', colornorm(label[-1]).cpu().detach(), global_step)
                    writer.add_image('img/pred', colornorm(res[-1]).cpu().detach(), global_step)
                    writer.add_image('img/mask', mask[-1].cpu().detach(), global_step)
                    writer.add_scalar('metric/psnr', metrics.psnr(res,label), global_step)
                    writer.add_scalar('metric/ssim', metrics.ssim(res,label), global_step)
            
            iter+=1
            global_step += 1
            loss_all+=loss.mean().item()
        
        endTime = time.time()
        print("epoch time is ",endTime - startTime)
        print("epoch %d mean loss for train is %f"%(e,loss_all/iter))

        if e > 20:
            scheduler.step()

        if e % 5 == 0:
            torch.save({'epoch': e + 1, 'state_dict': model.state_dict(), 
                        'optimizer': optimizer.state_dict()},
                         os.path.join(save_dir, 'totalModel.{}.pth.tar'.format(e)))

    torch.save(model.state_dict(), os.path.join(save_dir,modelSavePath))
 
if __name__ =="__main__":
    # Lewis
    dataset = get_Lewis_train_data()

    # Subway
    # dataset = get_Subway_train_data()

    # SunTemple
    # dataset = get_SunTemple_train_data()

    # Arena
    # dataset = get_Lewis_train_data() + get_Subway_train_data() + get_SunTemple_train_data()
    
    trainDiffuseLoader = data.DataLoader(dataset, batch_size, shuffle=True, num_workers=4, pin_memory=False)
    train(trainDiffuseLoader, "finalModel.pkl", 'checkpoints/Lewis', port=8097, reload=None)
   
