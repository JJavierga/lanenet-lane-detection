import cv2
import matplotlib as plt

import argparse
import os.path as ops

import os
from os import walk

import numpy as np

def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--image_path', type=str, help='The image path or the src image save dir')

    return parser.parse_args()


if __name__ == '__main__':
    """
    test code
    """
    # init args
    args = init_args()

    for f in os.listdir(args.image_path):
        if f.endswith('.png'):
            #newname=f[:-4]+'.png'
            #print(newname)
            img=cv2.imread(ops.join(args.image_path,f),cv2.IMREAD_GRAYSCALE)
            #print(img.shape)
            #cv2.imwrite(ops.join(args.image_path,f),img)
            #os.remove(ops.join(args.image_path,f))
            #os.rename(ops.join(args.image_path,f),ops.join(args.image_path,newname))

    

