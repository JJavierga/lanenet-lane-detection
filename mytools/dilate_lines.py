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


def get_files(image_path):
    """

    :input: image folder path
    :return: images' paths and new dirs
    """

    filenames = []
    dirnames = []
    new_dirs = []
    new_dirs_with_imgs = []
    
    for (_, dirname, filename) in walk(image_path):
        filenames.extend(filename)
        new_dirs_with_imgs.extend(filename)
        dirnames.extend(dirname)
        break

    os.makedirs(image_path+"_dilation_f", exist_ok=True)


    return filenames




def fill_lines(folder_path, subimg_path):
    """

    :input: image main path, images path new images path
    """

    complete_path=ops.join(folder_path, subimg_path)

    #print(complete_path)

    img=cv2.imread(complete_path,cv2.IMREAD_GRAYSCALE)
    
    #kernel_7 = np.ones((15,15),np.uint8)
    #kernel_2 = np.ones((2,2),np.uint8)
    #img=cv2.dilate(img,kernel_7,iterations=1)
    #img=cv2.erode(img,kernel_7,iterations=1)
    
    
    [fils,cols]=img.shape
    # Horizontal
    for fil in range(fils):
        for col in range(cols):
            if img[fil,col]==255:
                for i in range(2,3):
                    if(col+i < cols-1):
                        if img[fil, col+i]==255:
                            img[fil,col:col+i]=255*np.ones((1,i))
                            col=col+i
                            break
    
    # Vertical
    """for col in range(cols):
        for fil in range(fils):
            if img[fil,col]==255:
                for i in range(3,7):
                    if(fil+i < fils-1):
                        if img[fil+i, col]==255:
                            img[fil:fil+i,col]=255*np.ones((i))
                            fil=fil+i
                            break"""
    

    cv2.imwrite(ops.join(folder_path+"_dilation_f", subimg_path),img)
    #print(ops.join(folder_path+"_dilation_v_big", subimg_path))
     

if __name__ == '__main__':
    """
    test code
    """
    # init args
    args = init_args()

    file_list = get_files(args.image_path)

    for (files_number, file_name) in enumerate(file_list):
        #print(file_list[files_number])
        #print(new_dirs_list[files_number])

        fill_lines(args.image_path, file_list[files_number])

    




