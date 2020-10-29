import os
import cv2
import numpy as np
import PIL.Image as Image
import skimage.measure as ms

class Cal():
        
    def cal(self, root_dir):
        ssim = []
        psnr = []
        mat_files = open(root_dir,'r')
        for file_name in mat_files:
            gt_file = file_name.split(' ')[1][:-1]
            img_file =  file_name.split(' ')[0]
            print(gt_file)
            print(img_file)
            B = cv2.imread(gt_file)
            pred = cv2.imread(img_file)
            h,w=B.shape[:2]
            pred = cv2.resize(pred, (w,h))  
            pred = pred.astype(np.float32) / 255.0 
            B = B.astype(np.float32) / 255.0 
            pred = np.clip(pred, 0, 1)
            B = np.clip(B, 0, 1)    
            ssim.append(ms.compare_ssim(pred,B,multichannel=True))
            psnr.append(ms.compare_psnr(pred,B))
        return ssim,psnr

def run_test(root_dir):
    sess = Cal()
    ssim,psnr=sess.cal(root_dir)
    print("ssim:",np.mean(ssim),"psnr:",np.mean(psnr))

if __name__ == '__main__':
    root_dir="./test.txt"
    run_test(root_dir)

