import os
import argparse

from os.path import join

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--folders_path', type=str, help='Path to gt, binary and instance folders')

    return parser.parse_args()


if __name__ == '__main__':
    """
    Finds files in f1 that are not in f2
    """

    args = get_args()

    with open(join(args.folders_path,'train.txt'),'w') as f:
        gt_binary_folder=join(args.folders_path,'gt_binary_image')
        gt_instance_folder=join(args.folders_path,'gt_instance_image')
        gt_image_folder=join(args.folders_path,'gt_image')

        for file_name in os.listdir(gt_image_folder):
            binary_file_path=join(gt_binary_folder,file_name)
            instance_file_path=join(gt_instance_folder,file_name)
            image_file_path=join(gt_image_folder,file_name)

            info = '{:s} {:s} {:s}'.format(image_file_path, binary_file_path, instance_file_path)
            f.write(info + '\n')

