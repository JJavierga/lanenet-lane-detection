import cv2
import numpy as np

def Overlap_write(binary_seg, clustered_img, original_img, res_name='Res'):

    resized=cv2.resize(original_img,(256,512))
    
    resized=binary_seg * clustered_img + original_img * (1-binary_seg)

    cv2.imwrite('./Results/'+res_name+'.jpg', resized)

