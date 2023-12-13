import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "true"

import cv2
import tqdm
import numpy as np

from PIL import Image

def inference(save_dir, res_format, gt_path, min_id, max_id, offset):
    cnt = 0

    if gt_path is not None:
        gt_writer = None
    pred_writer = None

    for idx in tqdm.tqdm(range(min_id, max_id+1)):
        img = cv2.imread(os.path.join(save_dir,res_format%(idx+offset)))
        h,w,c = img.shape
        if h != 1080 or w != 1920:
            img = np.asarray(Image.fromarray(img).resize((1920, 1080), Image.BICUBIC))
        if gt_path is not None:
            gt = np.load(gt_path%idx).astype(np.float32)
            if img.shape != gt.shape:
                print(img.shape)
                print(gt.shape)
                print(f"Unmatched resolution at frame idx {idx}!!!")
                break

            gt = (np.clip(gt,0,1) * 255.0).astype(np.uint8)
        
            if gt_writer is None:
                h,w,c = gt.shape
                gt_writer = cv2.VideoWriter()
                gt_writer.open(os.path.join(save_dir, 'gt.avi'), cv2.VideoWriter_fourcc('p', 'n', 'g', ' '), 60, (w, h), True)
            gt_writer.write(gt)
        
        if pred_writer is None:
            h,w,c = img.shape
            pred_writer = cv2.VideoWriter()
            pred_writer.open(os.path.join(save_dir, 'pred.avi'), cv2.VideoWriter_fourcc('p', 'n', 'g', ' '), 60, (w, h), True)
        pred_writer.write(img)

        cnt += 1

    if gt_path is not None:
        gt_writer.release()
    pred_writer.release()

if __name__ == '__main__':
    inference('output/Lewis', 'res.%04d.png', '/home/user2/dataset/rendering/Lewis/test/HR/compressedHR.%04d.npy', 5, 1000, 0)