from mytools.bin2seg import bin2seg
from get_files import file_searcher

import cv2

import argparse
import os.path as ops

import os
from os import walk

#######
# python mytools/colour_lines2.py -i /home/javier/Pruebas/PLD-UAV/PLDU/train/aug_gt/Good/
#######


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

    new_dir = args.image_path + '_instance'

    os.makedirs(new_dir, exist_ok=True)

    converter=bin2seg()

    for file_name in os.listdir(args.image_path):
        
        img=cv2.imread(ops.join(args.image_path,file_name))
        print(ops.join(args.image_path,file_name))

        converter.reuse(img[:,:,1], 100)
        instance_img=converter.colour_lines()

        cv2.imwrite(ops.join(new_dir,file_name), instance_img)




    



