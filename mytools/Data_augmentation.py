import matplotlib as plt

import argparse
import os.path as ops

from get_files import file_searcher

import os
from os import walk

import cv2


### Don't use /home/, use instead /home

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

    Searcher=file_searcher(args.image_path)

    file_list=Searcher.get_files()

    for i in range(8):
        angle=i*45

        name0='_'+str(angle)+'_0'
        name1='_'+str(angle)+'_1'
    
        for file_name in file_list:

            print(file_name)

            comp_path=ops.join(args.image_path, file_name)
            ending=file_name[-4:]

            img=cv2.imread(comp_path)
            #print(ops.join(args.image_path,file_name))
            newname0=file_name[0:-4]+name0+ending
            #print(file_name[0:-4]+name0+ending)

            rotation_matrix = cv2.getRotationMatrix2D((img.shape[0] / 2, img.shape[1] / 2), angle, 1)
            img2 = cv2.warpAffine(img, rotation_matrix, (img.shape[0], img.shape[1]))

            cv2.imwrite(ops.join(args.image_path,newname0), img2) 
            #print(file_name[0:-5]+name1+ending)
            newname1=file_name[0:-4]+name1+ending

            img2=cv2.flip(img2, 1) # Horizontally

            cv2.imwrite(ops.join(args.image_path,newname1), img2) 



        