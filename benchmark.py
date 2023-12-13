import os
import time
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch_tensorrt

from thop import profile, clever_format
from model import STSSNet
from dataloaders import *

def benchmark(dataLoaderIns, model, half=False, use_trt=False, scale=1):
    model = model.to('cuda:0')
    model.eval()
    with torch.no_grad():
        for index, (input,features,mask, hisBuffer, label) in dataLoaderIns:
            input=input.cuda()
            hisBuffer=hisBuffer.cuda()
            mask=mask.cuda()
            features=features.cuda()

            B, C, H, W = input.shape

            if scale != 1:
                input = F.interpolate(input, scale_factor=scale, mode='bilinear')
                hisBuffer = F.interpolate(hisBuffer.reshape(B,-1,H,W), scale_factor=scale, mode='bilinear').reshape(3*B,-1,int(scale*H),int(scale*W))
                mask = F.interpolate(mask, scale_factor=scale, mode='bilinear')
                features = F.interpolate(features, scale_factor=scale, mode='bilinear')

            print('Input Shape:', list(input.shape[-2:]))

            B, C, H, W = input.shape
            input=F.pad(input,(0, 8-W%8,0,8-H%8))
            hisBuffer=F.pad(hisBuffer,(0, 8-W%8,0,8-H%8))
            mask=F.pad(mask,(0, 8-W%8,0,8-H%8))
            features=F.pad(features,(0, 8-W%8,0,8-H%8))   
            
            if half:
                model = model.half()
                input = input.half()
                hisBuffer = hisBuffer.half()
                mask = mask.half()
                features = features.half()

            # calculate flops
            macs, params = profile(model, inputs=(input, features, mask, hisBuffer))
            macs, params = clever_format([macs, params], "%.3f")
            print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))

            # compile trt model
            if use_trt:
                print('Compiling trt model...')
                traced_model = torch.jit.trace(model,(input, features, mask, hisBuffer))
                inputs = []
                for tensor in [input, features, mask, hisBuffer]:
                    inputs.append(torch_tensorrt.Input(list(tensor.shape), dtype=torch.half if half else torch.float))
                model = torch_tensorrt.compile(traced_model, 
                    inputs= inputs,
                    enabled_precisions= {torch.half if half else torch.float} # Run with FP16
                )
            else:
                print('Using traced model...')
                model = torch.jit.trace(model,(input, features, mask, hisBuffer))

            times = []

            # warm up
            for i in range(10):
              res=model(input, features, mask, hisBuffer)
            
            # benchmark
            for i in range(101):
              torch.cuda.synchronize()
              tt = time.time()
              res = model(input, features, mask, hisBuffer)
              torch.cuda.synchronize()
              times.append(time.time()-tt)

            print("Time: %.3f"%(np.mean(times[1:])*1000),'ms')
            break
 
if __name__ =="__main__":
    dataset = get_Lewis_test_data()
    testLoader = data.DataLoader(dataset,1,shuffle=False,num_workers=1, pin_memory=False)

    model = STSSNet(6,3,9,4)

    benchmark(testLoader, model, half=True, use_trt=True)
   