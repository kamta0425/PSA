# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 11:05:05 2017

@author: tanishka
"""
# import libraries
from __future__ import print_function
from __future__ import division

import numpy as np 
import pandas as pd
import os
import re

import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

import random
from timeit import default_timer as timer
import zone_division as zdiv


INPUT_FOLDER = 'stage1_aps'
PREPROCESSED_DATA_FOLDER = 'preprocessed/'
STAGE1_LABELS = 'stage1_labels.csv'
THREAT_ZONE = 1
BATCH_SIZE = 16
EXAMPLES_PER_SUBJECT = 182

FILE_LIST = []
TRAIN_TEST_SPLIT_RATIO = 0.2
TRAIN_SET_FILE_LIST = []
TEST_SET_FILE_LIST = []

IMAGE_DIM = 250
LEARNING_RATE = 1e-3
N_TRAIN_STEPS = 1
TRAIN_PATH = '/tsa_logs/train/'
MODEL_PATH = '/tsa_logs/model/'
MODEL_NAME = ('tsa-{}-lr-{}-{}-{}-tz-{}'.format('alexnet-v0.1', LEARNING_RATE, IMAGE_DIM, 
                                                IMAGE_DIM, THREAT_ZONE ))
def preprocess_tsa_data():
     # get a list of all subjects for whom there is data
    SUBJECT_LIST = [os.path.splitext(subject)[0] for subject in os.listdir(INPUT_FOLDER)]
    # intialize tracking and saving items
    batch_num = 1
    threat_zone_examples = []
    start_time = timer()
    
    for subject in SUBJECT_LIST:

        # read in the images
        print('--------------------------------------------------------------')
        print('t+> {:5.3f} |Reading images for subject #: {}'.format(timer()-start_time, 
                                                                     subject))
        print('--------------------------------------------------------------')
        images = zdiv.rdh.read_data(INPUT_FOLDER + '/' + subject + '.aps')

        # transpose so that the slice is the first dimension shape(16, 620, 512)
        images = images.transpose()

        # for each threat zone, loop through each image, mask off the zone and then crop it
        for tz_num, threat_zone_x_crop_dims in enumerate(zip(zdiv.zone_slice_list, 
                                                             zdiv.zone_crop_list)):

            threat_zone = threat_zone_x_crop_dims[0]
            crop_dims = threat_zone_x_crop_dims[1]

            # get label
            label = np.array(zdiv.zpd.get_subject_zone_label(tz_num, 
                             zdiv.zpd.get_subject_labels(STAGE1_LABELS, subject)))
            for tz_num, threat_zone_x_crop_dims in enumerate(zip(zdiv.zone_slice_list, 
                                                             zdiv.zone_crop_list)):

                threat_zone = threat_zone_x_crop_dims[0]
                crop_dims = threat_zone_x_crop_dims[1]
    
                # get label
                label = np.array(zdiv.zpd.get_subject_zone_label(tz_num, 
                                 zdiv.zpd.get_subject_labels(STAGE1_LABELS, subject)))
    
                for img_num, img in enumerate(images):
    
                    print('Threat Zone:Image -> {}:{}'.format(tz_num, img_num))
                    print('Threat Zone Label -> {}'.format(label))
                    
                    if threat_zone[img_num] is not None:
    
                        # correct the orientation of the image
                        print('-> reorienting base image') 
                        base_img = np.flipud(img)
                        print('-> shape {}|mean={}'.format(base_img.shape, 
                                                           base_img.mean()))
    
                        # convert to grayscale
                        print('-> converting to grayscale')
                        rescaled_img = zdiv.vimg.convert_to_grayscale(base_img)
                        print('-> shape {}|mean={}'.format(rescaled_img.shape, 
                                                           rescaled_img.mean()))
    
                        # spread the spectrum to improve contrast
                        print('-> spreading spectrum')
                        high_contrast_img = zdiv.vimg.spread_spectrum(rescaled_img)
                        print('-> shape {}|mean={}'.format(high_contrast_img.shape,
                                                           high_contrast_img.mean()))
    
                        # get the masked image
                        print('-> masking image')
                        masked_img = zdiv.vimg.roi(high_contrast_img, threat_zone[img_num])
                        print('-> shape {}|mean={}'.format(masked_img.shape, 
                                                           masked_img.mean()))
    
                        # crop the image
                        print('-> cropping image')
                        cropped_img = zdiv.vimg.crop(masked_img, crop_dims[img_num])
                        print('-> shape {}|mean={}'.format(cropped_img.shape, 
                                                           cropped_img.mean()))
    
                        # normalize the image
                        print('-> normalizing image')
                        normalized_img = zdiv.vimg.normalize(cropped_img)
                        print('-> shape {}|mean={}'.format(normalized_img.shape, 
                                                           normalized_img.mean()))
    
                        # zero center the image
                        print('-> zero centering')
                        zero_centered_img = zdiv.vimg.zero_center(normalized_img)
                        print('-> shape {}|mean={}'.format(zero_centered_img.shape, 
                                                           zero_centered_img.mean()))
    
                        # append the features and labels to this threat zone's example array
                        print ('-> appending example to threat zone {}'.format(tz_num))
                        threat_zone_examples.append([[tz_num], zero_centered_img, label])
                        print ('-> shape {:d}:{:d}:{:d}:{:d}:{:d}:{:d}'.format(
                                                             len(threat_zone_examples),
                                                             len(threat_zone_examples[0]),
                                                             len(threat_zone_examples[0][0]),
                                                             len(threat_zone_examples[0][1][0]),
                                                             len(threat_zone_examples[0][1][1]),
                                                             len(threat_zone_examples[0][2])))
                    else:
                        print('-> No view of tz:{} in img:{}. Skipping to next...'.format( 
                                    tz_num, img_num))
                    print('------------------------------------------------')
        # each subject gets EXAMPLES_PER_SUBJECT number of examples (182 to be exact, 
        # so this section just writes out the the data once there is a full minibatch 
        # complete.
        if ((len(threat_zone_examples) % (BATCH_SIZE * EXAMPLES_PER_SUBJECT)) == 0):
            for tz_num, tz in enumerate(zdiv.zone_slice_list):

                tz_examples_to_save = []

                # write out the batch and reset
                print(' -> writing: ' + PREPROCESSED_DATA_FOLDER + 
                                        'preprocessed_TSA_scans-tz{}-{}-{}-b{}.npy'.format( 
                                        tz_num+1,
                                        len(threat_zone_examples[0][1][0]),
                                        len(threat_zone_examples[0][1][1]), 
                                        batch_num))

                # get this tz's examples
                tz_examples = [example for example in threat_zone_examples if example[0] == 
                               [tz_num]]

                # drop unused columns
                tz_examples_to_save.append([[features_label[1], features_label[2]] 
                                            for features_label in tz_examples])

                # save batch.  Note that the trainer looks for tz{} where {} is a 
                # tz_num 1 based in the minibatch file to select which batches to 
                # use for training a given threat zone
                np.save(PREPROCESSED_DATA_FOLDER + 
                        'preprocessed_TSA_scans-tz{}-{}-{}-b{}.npy'.format(tz_num+1, 
                                                         len(threat_zone_examples[0][1][0]),
                                                         len(threat_zone_examples[0][1][1]), 
                                                         batch_num), 
                                                         tz_examples_to_save)
                del tz_examples_to_save

            #reset for next batch 
            del threat_zone_examples
            threat_zone_examples = []
            batch_num += 1
    
    # we may run out of subjects before we finish a batch, so we write out 
    # the last batch stub
    if (len(threat_zone_examples) > 0):
        for tz_num, tz in enumerate(zdiv.zone_slice_list):

            tz_examples_to_save = []

            # write out the batch and reset
            print(' -> writing: ' + PREPROCESSED_DATA_FOLDER 
                    + 'preprocessed_TSA_scans-tz{}-{}-{}-b{}.npy'.format(tz_num+1, 
                      len(threat_zone_examples[0][1][0]),
                      len(threat_zone_examples[0][1][1]), 
                                                                                                                  batch_num))
            # get this tz's examples
            tz_examples = [example for example in threat_zone_examples if example[0] == 
                           [tz_num]]

            # drop unused columns
            tz_examples_to_save.append([[features_label[1], features_label[2]] 
                                        for features_label in tz_examples])

            #save batch
            np.save(PREPROCESSED_DATA_FOLDER + 
                    'preprocessed_TSA_scans-tz{}-{}-{}-b{}.npy'.format(tz_num+1, 
                                                     len(threat_zone_examples[0][1][0]),
                                                     len(threat_zone_examples[0][1][1]), 
                                                     batch_num), 
                                                     tz_examples_to_save)
    
        