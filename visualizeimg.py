# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 13:36:09 2017

@author: tanishka
"""
import PSAreadingheader as psrd
from matplotlib import pyplot as plt
import cv2
import numpy as np
from scipy import stats
def plot_image_set(infile):

    # read in the aps file, it comes in as shape(512, 620, 16)
    img = psrd.read_data(infile)
    
    # transpose so that the slice is the first dimension shape(16, 620, 512)
    img = img.transpose()
        
    # show the graphs
    fig, axarr = plt.subplots(nrows=4, ncols=4, figsize=(10,10))
    
    i = 0
    for row in range(4):
        for col in range(4):
            resized_img = cv2.resize(img[i], (0,0), fx=0.1, fy=0.1)
            axarr[row, col].imshow(np.flipud(resized_img))
            i += 1
    
    print('Done!')

def get_single_image(infile, nth_image):

    # read in the aps file, it comes in as shape(512, 620, 16)
    img = psrd.read_data(infile)
    
    # transpose so that the slice is the first dimension shape(16, 620, 512)
    img = img.transpose()
    
    return np.flipud(img[nth_image])

def convert_to_grayscale(img):
    # scale pixel values to grayscale
    base_range = np.amax(img) - np.amin(img)
    rescaled_range = 255 - 0
    img_rescaled = (((img - np.amin(img)) * rescaled_range) / base_range)

    return np.uint8(img_rescaled)

def spread_spectrum(img):
    img = stats.threshold(img, threshmin=12, newval=0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img= clahe.apply(img)
    
    return img

def roi(img, vertices):
    
    # blank mask
    mask = np.zeros_like(img)

    # fill the mask
    cv2.fillPoly(mask, [vertices], 255)

    # now only show the area that is the mask
    masked = cv2.bitwise_and(img, mask)
    return masked

def crop(img, crop_list):

    x_coord = crop_list[0]
    y_coord = crop_list[1]
    width = crop_list[2]
    height = crop_list[3]
    cropped_img = img[x_coord:x_coord+width, y_coord:y_coord+height]
    
    return cropped_img

def normalize(image):
    MIN_BOUND = 0.0
    MAX_BOUND = 255.0
    
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image 

def zero_center(image):
     
    PIXEL_MEAN = 0.014327
    
    image = image - PIXEL_MEAN
    return image
   