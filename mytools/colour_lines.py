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
    parser.add_argument('--image_path', type=str, help='The image path or the src image save dir')

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
        dirnames.extend(dirname)
        break


    if dirnames:
        for dirname in dirnames:
            new_dirs.append(ops.join(image_path + "_instance", dirname))
            os.makedirs(new_dirs[-1], exist_ok=True)
            for (_, _, subfilenames) in walk(ops.join(image_path,dirname)):
                #print(type(dirname))
                #print(type(subfilenames))
                #print(subfilenames)
                for subfilename in subfilenames:
                    filenames.append(ops.join(dirname, subfilename))
                    new_dirs_with_imgs.append(ops.join(new_dirs[-1],subfilename))
                    #filenames.append(subfilename)
                break

    #print(filenames)
    #print(new_dirs)
    #print(new_dirs_with_imgs)

    return filenames, new_dirs, new_dirs_with_imgs 


def extend_colour(pos_f, pos_c, img, used, img_embedding, count):
    """

    :input: 
    """

    [fils, cols]=img.shape

    if (pos_f - 1 >= 0) and (pos_f + 1 < fils - 1) and (pos_c - 1 >= 0) and (pos_c + 1 < cols - 1):
        for i in range(-1,2):
            for j in range(-1,2):
                new_f = pos_f + i
                new_c = pos_c + j
                #print(new_f)
                #print(new_c)
                if(img[new_f, new_c]==255 and used[new_f, new_c]==False):
                    #print('G')
                    used[new_f, new_c]=True
                    img_embedding[new_f, new_c]= count*50 + 20
                    extend_colour(new_f, new_c, img, used, img_embedding, count)

    return img_embedding, used



def colour_lines(folder_path, subimg_path, new_img_path):
    """

    :input: image main path, images path new images path
    """

    complete_path=ops.join(folder_path, subimg_path)

    #print(complete_path)

    img=cv2.imread(complete_path,cv2.IMREAD_GRAYSCALE)

    [fils, cols]=img.shape

    img_embedding=img
    used=np.zeros((fils,cols))

    count=1
    for col in range(cols):
        for fil in range(fils):
            if img[fil,col]==255 and used[fil,col]==False:
                #print('F')
                img_embedding, used = extend_colour(fil, col, img, used, img_embedding, count)
                count+=1

    #print(new_img_path)
    #print(folder_path)
    #print(subimg_path)
    cv2.imwrite(new_img_path,img_embedding)
     

if __name__ == '__main__':
    """
    test code
    """
    # init args
    args = init_args()

    file_list, new_dirs_list, new_dirs_with_imgs = get_files(args.image_path)

    for (files_number, file_name) in enumerate(file_list):
        #print(file_list[files_number])
        #print(new_dirs_list[files_number])

        colour_lines(args.image_path, file_list[files_number], new_dirs_with_imgs[files_number])

    




