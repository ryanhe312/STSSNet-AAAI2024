import torch.utils.data as data
import torch
import torch.nn.functional as F
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "true"
import cv2
import numpy as np
import time
import random
import pickle
import torchvision as tv


def ReadData(path, rd=None, depth_op = 'clip'):

    try:
        total = np.load(path)
    except:
        print(path)
        quit()

    if rd is not None:
        if rd<0.2:
            total = np.flip(total,0)
        elif rd<0.3:
            total = np.flip(total, 1)
        elif rd<0.35:
            total = np.flip(total,(0,1))
    
    img = total[:,:,0:3]
    img3 = total[:,:,3:6]
    img5 = total[:,:,6:9]
    imgOcc = total[:,:,9:12]
    img_no_hole = total[:,:,12:15]
    img_no_hole3 = total[:,:,15:18]
    img_no_hole5 = total[:,:,18:21]
    basecolor = total[:,:,21:24]
    metalic = total[:,:,24:25]
    roughness = total[:,:,25:26]
    depth = total[:,:,26:27]
    normal = total[:,:,27:30]
    
    if total.shape[2] == 35:
        motion = total[:,:,31:34]
    else:
        motion = total[:,:,30:33]
    img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2BGR)
    img3 = cv2.cvtColor(img3.astype(np.float32), cv2.COLOR_RGB2BGR)
    img5 = cv2.cvtColor(img5.astype(np.float32), cv2.COLOR_RGB2BGR)
    imgOcc = cv2.cvtColor(imgOcc.astype(np.float32), cv2.COLOR_RGB2BGR)
    img_no_hole = cv2.cvtColor(img_no_hole.astype(np.float32), cv2.COLOR_RGB2BGR)
    img_no_hole3 = cv2.cvtColor(img_no_hole3.astype(np.float32), cv2.COLOR_RGB2BGR)
    img_no_hole5 = cv2.cvtColor(img_no_hole5.astype(np.float32), cv2.COLOR_RGB2BGR)
    basecolor = cv2.cvtColor(basecolor.astype(np.float32), cv2.COLOR_RGB2BGR)
    normal = cv2.cvtColor(normal.astype(np.float32), cv2.COLOR_RGB2BGR)

    if depth_op == 'clip':
        depth = np.clip(depth,0,1)
    else:
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)

    motion = motion.astype(np.float32)
    motion = cv2.cvtColor(motion, cv2.COLOR_BGR2RGB)[:,:,:2]
    motion[:,:,0] = -motion[:,:,0]

    return img, img3, img5, imgOcc, img_no_hole, img_no_hole3, img_no_hole5, basecolor, metalic, roughness, depth, normal, motion


def crop(lst,gt,size=256):
    i, j, h, w = tv.transforms.RandomCrop.get_params(lst[0], output_size=(size, size))
    for i in range(len(lst)):
        lst[i] = tv.transforms.functional.crop(lst[i], i, j, h, w)
    gt = tv.transforms.functional.crop(gt, i*2, j*2, h*2, w*2)
    return lst, gt

def form_input(data_path, is_train, index, get_his=True, get_motion=False, add_mask=True, random_flip=False, depth_op='clip'):

    npz_path, hr_path = data_path

    if is_train and random_flip:
        rd = random.random()
    else:
        rd = None

    gt = np.load(hr_path)
    gt = cv2.cvtColor(gt.astype(np.float32), cv2.COLOR_RGB2BGR)

    img, img_2, img_3, occ_warp_img, woCheckimg, woCheckimg_2, woCheckimg_3, basecolor, metalic, Roughnessimg, Depthimg, Normalimg, motion = ReadData(npz_path, rd, depth_op)

    input = img
    mask = input.copy()
    mask[mask==0.]=1.0
    mask[mask==-1]=0.0
    mask[mask!=0.0]=1.0

    occ_warp_img[occ_warp_img < 0.0] = 0.0
    woCheckimg[woCheckimg < 0.0] = 0.0

    features = np.concatenate([Normalimg, Depthimg, Roughnessimg, metalic, basecolor], axis=2).copy()

    if get_his:
        woCheckimg_2[woCheckimg_2 < 0.0] = 0.0
        woCheckimg_3[woCheckimg_3 < 0.0] = 0.0
        mask2 = img_2.copy()
        mask2[mask2==0.]=1.0
        mask2[mask2==-1]=0.0
        mask2[mask2!=0.0]=1.0

        mask3 = img_3.copy()
        mask3[mask3==0.]=1.0
        mask3[mask3==-1]=0.0
        mask3[mask3!=0.0]=1.0

        his_1 = np.concatenate([woCheckimg, mask[:,:,0].reshape((Normalimg.shape[0],Normalimg.shape[1], 1))], axis=2).transpose([2,0,1]).reshape(1, 4, Normalimg.shape[0],Normalimg.shape[1])
        his_2 = np.concatenate([woCheckimg_2, mask2[:,:,0].reshape((Normalimg.shape[0],Normalimg.shape[1], 1))], axis=2).transpose([2,0,1]).reshape(1, 4, Normalimg.shape[0],Normalimg.shape[1])
        his_3 = np.concatenate([woCheckimg_3, mask3[:,:,0].reshape((Normalimg.shape[0],Normalimg.shape[1], 1))], axis=2).transpose([2,0,1]).reshape(1, 4, Normalimg.shape[0],Normalimg.shape[1])
        hisBuffer = np.concatenate([his_3, his_2, his_1], axis=0)
    
    input = np.concatenate([occ_warp_img,woCheckimg],axis=2).copy()
    
    if is_train and add_mask:
        h,w,c = input.shape
        
        h0 = random.randint(1,h//4)
        w0 = random.randint(1,min(h*w//16//h0, w//4))
        
        hstart = random.randint(0, h-h0)
        wstart = random.randint(0, w-w0)
        
        full_mask = np.zeros((h,w,3),dtype=np.uint8)
        cv2.rectangle(full_mask, (wstart,hstart), (wstart+w0-1,hstart+h0-1), (255, 255, 255), -1)
        
        mask = np.logical_and((full_mask[:,:,:1] == 0), mask[:,:,:1]).astype(np.float32)
    else:
        mask = mask[:,:,:1].astype(np.float32)
    
    if rd is not None:
        if rd<0.2:
            gt = np.flip(gt,0)
        elif rd<0.3:
            gt = np.flip(gt, 1)
        elif rd<0.35:
            gt = np.flip(gt,(0,1))

    out_puts = []
    out_puts.extend([torch.tensor(input.transpose([2,0,1])), torch.tensor(features.transpose([2,0,1])), torch.tensor(mask.transpose([2,0,1]))])
    if get_his:
        out_puts.append(torch.tensor(hisBuffer))
    out_puts.append(torch.tensor(gt.copy().transpose([2,0,1])))
    if get_motion:
        out_puts.append(torch.tensor(motion.transpose([2,0,1])))

    return out_puts
        
class RenderDataset(data.Dataset):
    def __init__(self, npz_formats, HR_format, idx_list, is_train=True, depth_op='clip'):  
        self.depth_op = depth_op
        self.is_train = is_train
        self.data_list = []
        
        for npz_format in npz_formats:
            for idx in idx_list:
                if idx == 1427: continue
                npz_path = npz_format%idx
                hr_path = HR_format%idx
                if os.path.exists(npz_path) and os.path.exists(hr_path):
                    self.data_list.append((npz_path, hr_path))
            
    def __getitem__(self, index):
        outputs = form_input(self.data_list[index], self.is_train, index, get_his=True, random_flip=False, depth_op=self.depth_op)
        return outputs
    
    def __len__(self):
        return len(self.data_list)
    
class RenderDataset_eval(data.Dataset):
    def __init__(self, SF_format, IF_format, HR_format, idx_list, is_train=False, depth_op='clip'):  
        self.depth_op = depth_op
        self.data_list = []
        self.is_train = is_train
        
        for idx in idx_list:
            if idx % 2 == 0:
                npz_path = SF_format%idx
            else:
                npz_path = IF_format%idx
            hr_path = HR_format%idx

            if os.path.exists(npz_path) and os.path.exists(hr_path):
                self.data_list.append((idx, npz_path, hr_path))
            else:
                print(idx)
                print(npz_path)
                print(hr_path)
                quit()
            
    def __getitem__(self, index):
        outputs = form_input(self.data_list[index][1:], False, index, get_motion=False, add_mask=False, depth_op = self.depth_op)

        if self.is_train:
            return outputs
        else:
            return torch.tensor(self.data_list[index][0]), outputs
    
    def __len__(self):
        return len(self.data_list)
    
def get_SunTemple_train_data():
    warp_format = '/home/user2/dataset/rendering/SunTemple/train1/NPY/compressed.%04d.Warp.npy'
    nowarp_format = '/home/user2/dataset/rendering/SunTemple/train1/NPY/compressed.%04d.NoWarp.npy'
    HR_format = '/home/user2/dataset/rendering/SunTemple/train1/HR/compressedHR.%04d.npy'
    idx_list = list(range(5,3002))
    dataset1 = RenderDataset([warp_format, nowarp_format], HR_format, idx_list)
    print(len(dataset1))
    warp_format = '/home/user2/dataset/rendering/SunTemple/train2/NPY/compressed.%04d.Warp.npy'
    nowarp_format = '/home/user2/dataset/rendering/SunTemple/train2/NPY/compressed.%04d.NoWarp.npy'
    HR_format = '/home/user2/dataset/rendering/SunTemple/train2/HR/compressedHR.%04d.npy'
    idx_list = list(range(5,2747))
    dataset2 = RenderDataset([warp_format, nowarp_format], HR_format, idx_list)
    print(len(dataset2))
    
    return dataset1 + dataset2

def get_Subway_train_data():
    warp_format = '/home/user2/dataset/rendering/Subway/training1/NPY/compressed.%04d.Warp.npy'
    nowarp_format = '/home/user2/dataset/rendering/Subway/training1/NPY/compressed.%04d.NoWarp.npy'
    HR_format = '/home/user2/dataset/rendering/Subway/training1/HR/compressedHR.%04d.npy'
    idx_list = list(range(6,3005))
    dataset1 = RenderDataset([warp_format, nowarp_format], HR_format, idx_list)
    print(len(dataset1))
    
    warp_format = '/home/user2/dataset/rendering/Subway/training2/NPY/compressed.%04d.Warp.npy'
    nowarp_format = '/home/user2/dataset/rendering/Subway/training2/NPY/compressed.%04d.NoWarp.npy'
    HR_format = '/home/user2/dataset/rendering/Subway/training2/HR/compressedHR.%04d.npy'
    idx_list = list(range(6,3600))
    dataset2 = RenderDataset([warp_format, nowarp_format], HR_format, idx_list)
    print(len(dataset2))
    
    return dataset1 + dataset2

def get_Lewis_train_data():
    warp_format = '/home/user2/dataset/rendering/Lewis/training1/NPY/compressed.%04d.Warp.npy'
    nowarp_format = '/home/user2/dataset/rendering/Lewis/training1/NPY/compressed.%04d.NoWarp.npy'
    HR_format = '/home/user2/dataset/rendering/Lewis/training1/HR/compressedHR.%04d.npy'
    idx_list = list(range(9, 2803))
    dataset1 = RenderDataset([warp_format, nowarp_format], HR_format, idx_list, depth_op='scale')
    print(len(dataset1))
    
    warp_format = '/home/user2/dataset/rendering/Lewis/training2/NPY/compressed.%04d.Warp.npy'
    nowarp_format = '/home/user2/dataset/rendering/Lewis/training2/NPY/compressed.%04d.NoWarp.npy'
    HR_format = '/home/user2/dataset/rendering/Lewis/training2/HR/compressedHR.%04d.npy'
    idx_list = list(range(5,3585))
    dataset2 = RenderDataset([warp_format, nowarp_format], HR_format, idx_list, depth_op='scale')
    print(len(dataset2))
    
    return dataset1 + dataset2

def get_Lewis_test_data(is_train=False, depth_op='scale'):
    warp_format = '/home/user2/dataset/rendering/Lewis/test/NPY/compressed.%04d.Warp.npy'
    nowarp_format = '/home/user2/dataset/rendering/Lewis/test/NPY/compressed.%04d.NoWarp.npy'
    HR_format = '/home/user2/dataset/rendering/Lewis/test/HR/compressedHR.%04d.npy'
    idx_list = list(range(5,1001))
    return RenderDataset_eval(nowarp_format, warp_format, HR_format, idx_list, is_train=is_train, depth_op=depth_op)

def get_Subway_test_data(is_train=False):
    warp_format = '/home/user2/dataset/rendering/Subway/test/NPY/compressed.%04d.Warp.npy'
    nowarp_format = '/home/user2/dataset/rendering/Subway/test/NPY/compressed.%04d.NoWarp.npy'
    HR_format = '/home/user2/dataset/rendering/Subway/test/HR/compressedHR.%04d.npy'
    idx_list = list(range(6,1001))
    return RenderDataset_eval(nowarp_format, warp_format, HR_format, idx_list, is_train=is_train)

def get_Arena_test_data(is_train=False):
    warp_format = '/home/user2/dataset/rendering/Arena/test/NPY/compressed.%04d.Warp.npy'
    nowarp_format = '/home/user2/dataset/rendering/Arena/test/NPY/compressed.%04d.NoWarp.npy'
    HR_format = '/home/user2/dataset/rendering/Arena/test/HR/compressedHR.%04d.npy'
    idx_list = list(range(500,1501))
    return RenderDataset_eval(nowarp_format, warp_format, HR_format, idx_list, is_train=is_train)

def get_SunTemple_test_data(is_train=False):
    warp_format = '/home/user2/dataset/rendering/SunTemple/test/NPY/compressed.%04d.Warp.npy'
    nowarp_format = '/home/user2/dataset/rendering/SunTemple/test/NPY/compressed.%04d.NoWarp.npy'
    HR_format = '/home/user2/dataset/rendering/SunTemple/test/HR/compressedHR.%04d.npy'
    idx_list = list(range(5,1001))
    return RenderDataset_eval(nowarp_format, warp_format, HR_format, idx_list, is_train=is_train)

if __name__ == '__main__':
    dataset = get_Lewis_train_data()
    print(len(dataset))
    input,features,mask,hisBuffer,label = dataset[0]
    print(input.shape,features.shape,mask.shape,hisBuffer.shape,label.shape)