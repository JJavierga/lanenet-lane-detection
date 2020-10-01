#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-23 上午11:33
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : test_lanenet.py
# @IDE: PyCharm Community Edition
# Edited
"""
test LaneNet model on single image
"""

######
# python tools/test_lanenet_video_frames.py --weights_path ./weights/tusimple_lanenet.ckpt --video_path /home/javier/Programming/Segmentation/ENet/ENet-Real-Time-Semantic-Segmentation/datasets/cars.mp4
######

import argparse
import os.path as ops
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from local_utils.config_utils import parse_config_utils
from local_utils.log_util import init_logger

from os import walk

CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_test')


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help='The videp path or the src video save dir')
    parser.add_argument('--weights_path', type=str, help='The model weights path')

    return parser.parse_args()


def args_str2bool(arg_value):
    """

    :param arg_value:
    :return:
    """
    if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def test_lanenet(video_path, weights_path):
    """

    :param video_path:
    :param weights_path:
    :return:
    """
    assert ops.exists(video_path), '{:s} not exist'.format(video_path)

    #####

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

    net = lanenet.LaneNet(phase='test', cfg=CFG)
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='LaneNet')
    
    postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)

    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    # define moving average version of the learned variables for eval
    with tf.variable_scope(name_or_scope='moving_avg'):
        variable_averages = tf.train.ExponentialMovingAverage(
            CFG.SOLVER.MOVING_AVE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()

    # define saver
    saver = tf.train.Saver(variables_to_restore)

    #####

    f = []

    for (dirpath, dirnames, filenames) in walk(video_path):
        f.extend(filenames)

    #####
    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)
    #####

        for img_name in f:

            LOG.info('Start reading image and preprocessing')
            t_start = time.time()
            frame=cv2.imread(video_path+img_name)


            image_vis = frame
            image = cv2.resize(frame, (512, 256), interpolation=cv2.INTER_LINEAR)
            image = image / 127.5 - 1.0
            LOG.info('Image load complete, cost time: {:.5f}s'.format(time.time() - t_start))


            t_start = time.time()
            loop_times = 500
            for i in range(loop_times):
                binary_seg_image, instance_seg_image = sess.run(
                    [binary_seg_ret, instance_seg_ret],
                    feed_dict={input_tensor: [image]}
                )
            t_cost = time.time() - t_start
            t_cost /= loop_times
            LOG.info('Single imgae inference cost time: {:.5f}s'.format(t_cost))

            postprocess_result = postprocessor.postprocess(
                binary_seg_result=binary_seg_image[0],
                instance_seg_result=instance_seg_image[0],
                source_image=image_vis
            )
            mask_image = postprocess_result['mask_image']

            for i in range(CFG.MODEL.EMBEDDING_FEATS_DIMS):
                instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
            embedding_image = np.array(instance_seg_image[0], np.uint8)


            cv2.imwrite('./Results/'+'post'+img_name, image_vis[:, :, (0, 1, 2)])
                

    sess.close()

    return


if __name__ == '__main__':
    """
    test code
    """
    # init args
    args = init_args()

    test_lanenet(args.video_path, args.weights_path)
