import argparse
import os.path as ops

from get_files import file_searcher

import os
from os import walk

import shutil


### Don't use /home/, use instead /home

def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-a','--all_data_path', type=str, help='The image path or the src image save dir')
    parser.add_argument('-e','--entangled_files_path', type=str, help='The image path or the src image save dir')
    parser.add_argument('-p','--paste_path', type=str, help='The image path or the src image save dir')


    return parser.parse_args()



if __name__ == '__main__':
    """
    test code
    """
    # init args
    args = init_args()

    Searcher=file_searcher(args.entangled_files_path)
    file_list=Searcher.get_files()

    Searcher=file_searcher(args.all_data_path)
    all_file_list=Searcher.get_files()


    flag=0
    for file_name in file_list:
        for missing_file_name in all_file_list:
            if(file_name==missing_file_name):
                flag=1
                break
        if flag==1:
            shutil.copy(ops.join(args.all_data_path, file_name), args.paste_path)
        flag=0
