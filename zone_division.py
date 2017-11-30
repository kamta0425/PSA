# -*- coding: utf-8 -*-

"""
Created on Mon Oct 23 13:25:59 2017

@author: KAMMO, TANNY, AJ, SUSHI, GOWTHAMI
"""

# imports
from __future__ import print_function
from __future__ import division
import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import PSAreadingheader as rdh
import zone_probablity as zpd
import visualizeimg as vimg

COLORMAP = 'pink'
APS_FILE_NAME = 'stage1_aps/00360f79fd6e02781457eda48f85da90.aps'
BODY_ZONES = 'body_zones.png'
THREAT_LABELS = 'stage1_labels.csv'

# show the threat zones
body_zones_img = plt.imread(BODY_ZONES)
fig, ax = plt.subplots(figsize=(15,15))
#ax.imshow(body_zones_img)

sector01_pts = np.array([[0,160],[200,160],[200,230],[0,230]], np.int32)
sector02_pts = np.array([[0,0],[200,0],[200,160],[0,160]], np.int32)
sector03_pts = np.array([[330,160],[512,160],[512,240],[330,240]], np.int32)
sector04_pts = np.array([[350,0],[512,0],[512,160],[350,160]], np.int32)

# sector 5 is used for both threat zone 5 and 17
sector05_pts = np.array([[0,220],[512,220],[512,300],[0,300]], np.int32) 

sector06_pts = np.array([[0,300],[256,300],[256,360],[0,360]], np.int32)
sector07_pts = np.array([[256,300],[512,300],[512,360],[256,360]], np.int32)
sector08_pts = np.array([[0,370],[225,370],[225,450],[0,450]], np.int32)
sector09_pts = np.array([[225,370],[275,370],[275,450],[225,450]], np.int32)
sector10_pts = np.array([[275,370],[512,370],[512,450],[275,450]], np.int32)
sector11_pts = np.array([[0,450],[256,450],[256,525],[0,525]], np.int32)
sector12_pts = np.array([[256,450],[512,450],[512,525],[256,525]], np.int32)
sector13_pts = np.array([[0,525],[256,525],[256,600],[0,600]], np.int32)
sector14_pts = np.array([[256,525],[512,525],[512,600],[256,600]], np.int32)
sector15_pts = np.array([[0,600],[256,600],[256,660],[0,660]], np.int32)
sector16_pts = np.array([[256,600],[512,600],[512,660],[256,660]], np.int32)

# crop dimensions, upper left x, y, width, height
sector_crop_list = [[ 50,  50, 250, 250], # sector 1
                    [  0,   0, 250, 250], # sector 2
                    [ 50, 250, 250, 250], # sector 3
                    [250,   0, 250, 250], # sector 4
                    [150, 150, 250, 250], # sector 5/17
                    [200, 100, 250, 250], # sector 6
                    [200, 150, 250, 250], # sector 7
                    [250,  50, 250, 250], # sector 8
                    [250, 150, 250, 250], # sector 9
                    [300, 200, 250, 250], # sector 10
                    [400, 100, 250, 250], # sector 11
                    [350, 200, 250, 250], # sector 12
                    [410,   0, 250, 250], # sector 13
                    [410, 200, 250, 250], # sector 14
                    [410,   0, 250, 250], # sector 15
                    [410, 200, 250, 250], # sector 16
                   ]


# Each element in the zone_slice_list contains the sector to use in the call to roi()
zone_slice_list = [ [ # threat zone 1
                      sector01_pts, sector01_pts, sector01_pts, None, 
                      None, None, sector03_pts, sector03_pts, 
                      sector03_pts, sector03_pts, sector03_pts, 
                      None, None, sector01_pts, sector01_pts, sector01_pts ], 
    
                    [ # threat zone 2
                      sector02_pts, sector02_pts, sector02_pts, None, 
                      None, None, sector04_pts, sector04_pts, 
                      sector04_pts, sector04_pts, sector04_pts, None, 
                      None, sector02_pts, sector02_pts, sector02_pts ],
    
                    [ # threat zone 3
                      sector03_pts, sector03_pts, sector03_pts, sector03_pts, 
                      None, None, sector01_pts, sector01_pts,
                      sector01_pts, sector01_pts, sector01_pts, sector01_pts, 
                      None, None, sector03_pts, sector03_pts ],
    
                    [ # threat zone 4
                      sector04_pts, sector04_pts, sector04_pts, sector04_pts, 
                      None, None, sector02_pts, sector02_pts, 
                      sector02_pts, sector02_pts, sector02_pts, sector02_pts, 
                      None, None, sector04_pts, sector04_pts ],
    
                    [ # threat zone 5
                      sector05_pts, sector05_pts, sector05_pts, sector05_pts, 
                      sector05_pts, sector05_pts, sector05_pts, sector05_pts,
                      None, None, None, None, 
                      None, None, None, None ],
    
                    [ # threat zone 6
                      sector06_pts, None, None, None, 
                      None, None, None, None, 
                      sector07_pts, sector07_pts, sector06_pts, sector06_pts, 
                      sector06_pts, sector06_pts, sector06_pts, sector06_pts ],
    
                    [ # threat zone 7
                      sector07_pts, sector07_pts, sector07_pts, sector07_pts, 
                      sector07_pts, sector07_pts, sector07_pts, sector07_pts, 
                      None, None, None, None, 
                      None, None, None, None ],
    
                    [ # threat zone 8
                      sector08_pts, sector08_pts, None, None, 
                      None, None, None, sector10_pts, 
                      sector10_pts, sector10_pts, sector10_pts, sector10_pts, 
                      sector08_pts, sector08_pts, sector08_pts, sector08_pts ],
    
                    [ # threat zone 9
                      sector09_pts, sector09_pts, sector08_pts, sector08_pts, 
                      sector08_pts, None, None, None,
                      sector09_pts, sector09_pts, None, None, 
                      None, None, sector10_pts, sector09_pts ],
    
                    [ # threat zone 10
                      sector10_pts, sector10_pts, sector10_pts, sector10_pts, 
                      sector10_pts, sector08_pts, sector10_pts, None, 
                      None, None, None, None, 
                      None, None, None, sector10_pts ],
    
                    [ # threat zone 11
                      sector11_pts, sector11_pts, sector11_pts, sector11_pts, 
                      None, None, sector12_pts, sector12_pts,
                      sector12_pts, sector12_pts, sector12_pts, None, 
                      sector11_pts, sector11_pts, sector11_pts, sector11_pts ],
    
                    [ # threat zone 12
                      sector12_pts, sector12_pts, sector12_pts, sector12_pts, 
                      sector12_pts, sector11_pts, sector11_pts, sector11_pts, 
                      sector11_pts, sector11_pts, sector11_pts, None, 
                      None, sector12_pts, sector12_pts, sector12_pts ],
    
                    [ # threat zone 13
                      sector13_pts, sector13_pts, sector13_pts, sector13_pts, 
                      None, None, sector14_pts, sector14_pts,
                      sector14_pts, sector14_pts, sector14_pts, None, 
                      sector13_pts, sector13_pts, sector13_pts, sector13_pts ],
    
                    [ # sector 14
                      sector14_pts, sector14_pts, sector14_pts, sector14_pts, 
                      sector14_pts, None, sector13_pts, sector13_pts, 
                      sector13_pts, sector13_pts, sector13_pts, None, 
                      None, None, None, None ],
    
                    [ # threat zone 15
                      sector15_pts, sector15_pts, sector15_pts, sector15_pts, 
                      None, None, sector16_pts, sector16_pts,
                      sector16_pts, sector16_pts, None, sector15_pts, 
                      sector15_pts, None, sector15_pts, sector15_pts ],
    
                    [ # threat zone 16
                      sector16_pts, sector16_pts, sector16_pts, sector16_pts, 
                      sector16_pts, sector16_pts, sector15_pts, sector15_pts, 
                      sector15_pts, sector15_pts, sector15_pts, None, 
                      None, None, sector16_pts, sector16_pts ],
    
                    [ # threat zone 17
                      None, None, None, None, 
                      None, None, None, None,
                      sector05_pts, sector05_pts, sector05_pts, sector05_pts, 
                      sector05_pts, sector05_pts, sector05_pts, sector05_pts ] ]

# Each element in the zone_slice_list contains the sector to use in the call to roi()
zone_crop_list =  [ [ # threat zone 1
                      sector_crop_list[0], sector_crop_list[0], sector_crop_list[0], None, 
                      None, None, sector_crop_list[2], sector_crop_list[2], 
                      sector_crop_list[2], sector_crop_list[2], sector_crop_list[2], None, 
                      None, sector_crop_list[0], sector_crop_list[0], 
                      sector_crop_list[0] ],
    
                    [ # threat zone 2
                      sector_crop_list[1], sector_crop_list[1], sector_crop_list[1], None, 
                      None, None, sector_crop_list[3], sector_crop_list[3], 
                      sector_crop_list[3], sector_crop_list[3], sector_crop_list[3], 
                      None, None, sector_crop_list[1], sector_crop_list[1], 
                      sector_crop_list[1] ],
    
                    [ # threat zone 3
                      sector_crop_list[2], sector_crop_list[2], sector_crop_list[2], 
                      sector_crop_list[2], None, None, sector_crop_list[0], 
                      sector_crop_list[0], sector_crop_list[0], sector_crop_list[0], 
                      sector_crop_list[0], sector_crop_list[0], None, None, 
                      sector_crop_list[2], sector_crop_list[2] ],
               
                    [ # threat zone 4
                      sector_crop_list[3], sector_crop_list[3], sector_crop_list[3], 
                      sector_crop_list[3], None, None, sector_crop_list[1], 
                      sector_crop_list[1], sector_crop_list[1], sector_crop_list[1], 
                      sector_crop_list[1], sector_crop_list[1], None, None, 
                      sector_crop_list[3], sector_crop_list[3] ],
                    
                    [ # threat zone 5
                      sector_crop_list[4], sector_crop_list[4], sector_crop_list[4], 
                      sector_crop_list[4], sector_crop_list[4], sector_crop_list[4], 
                      sector_crop_list[4], sector_crop_list[4],
                      None, None, None, None, None, None, None, None ],
                    
                    [ # threat zone 6
                      sector_crop_list[5], None, None, None, None, None, None, None, 
                      sector_crop_list[6], sector_crop_list[6], sector_crop_list[5], 
                      sector_crop_list[5], sector_crop_list[5], sector_crop_list[5], 
                      sector_crop_list[5], sector_crop_list[5] ],
    
                    [ # threat zone 7
                      sector_crop_list[6], sector_crop_list[6], sector_crop_list[6], 
                      sector_crop_list[6], sector_crop_list[6], sector_crop_list[6], 
                      sector_crop_list[6], sector_crop_list[6], 
                      None, None, None, None, None, None, None, None ],
    
                    [ # threat zone 8
                      sector_crop_list[7], sector_crop_list[7], None, None, None, 
                      None, None, sector_crop_list[9], sector_crop_list[9], 
                      sector_crop_list[9], sector_crop_list[9], sector_crop_list[9], 
                      sector_crop_list[7], sector_crop_list[7], sector_crop_list[7], 
                      sector_crop_list[7] ],
    
                    [ # threat zone 9
                      sector_crop_list[8], sector_crop_list[8], sector_crop_list[7], 
                      sector_crop_list[7], sector_crop_list[7], None, None, None,
                      sector_crop_list[8], sector_crop_list[8], None, None, None, 
                      None, sector_crop_list[9], sector_crop_list[8] ],
    
                    [ # threat zone 10
                      sector_crop_list[9], sector_crop_list[9], sector_crop_list[9], 
                      sector_crop_list[9], sector_crop_list[9], sector_crop_list[7], 
                      sector_crop_list[9], None, None, None, None, None, None, None, 
                      None, sector_crop_list[9] ],
    
                    [ # threat zone 11
                      sector_crop_list[10], sector_crop_list[10], sector_crop_list[10], 
                      sector_crop_list[10], None, None, sector_crop_list[11], 
                      sector_crop_list[11], sector_crop_list[11], sector_crop_list[11], 
                      sector_crop_list[11], None, sector_crop_list[10], 
                      sector_crop_list[10], sector_crop_list[10], sector_crop_list[10] ],
    
                    [ # threat zone 12
                      sector_crop_list[11], sector_crop_list[11], sector_crop_list[11], 
                      sector_crop_list[11], sector_crop_list[11], sector_crop_list[11], 
                      sector_crop_list[11], sector_crop_list[11], sector_crop_list[11], 
                      sector_crop_list[11], sector_crop_list[11], None, None, 
                      sector_crop_list[11], sector_crop_list[11], sector_crop_list[11] ],
    
                    [ # threat zone 13
                      sector_crop_list[12], sector_crop_list[12], sector_crop_list[12], 
                      sector_crop_list[12], None, None, sector_crop_list[13], 
                      sector_crop_list[13], sector_crop_list[13], sector_crop_list[13], 
                      sector_crop_list[13], None, sector_crop_list[12], 
                      sector_crop_list[12], sector_crop_list[12], sector_crop_list[12] ],
    
                    [ # sector 14
                      sector_crop_list[13], sector_crop_list[13], sector_crop_list[13], 
                      sector_crop_list[13], sector_crop_list[13], None, 
                      sector_crop_list[13], sector_crop_list[13], sector_crop_list[12], 
                      sector_crop_list[12], sector_crop_list[12], None, None, None, 
                      None, None ],
    
                    [ # threat zone 15
                      sector_crop_list[14], sector_crop_list[14], sector_crop_list[14], 
                      sector_crop_list[14], None, None, sector_crop_list[15], 
                      sector_crop_list[15], sector_crop_list[15], sector_crop_list[15], 
                      None, sector_crop_list[14], sector_crop_list[14], None, 
                      sector_crop_list[14], sector_crop_list[14] ],
    
                    [ # threat zone 16
                      sector_crop_list[15], sector_crop_list[15], sector_crop_list[15], 
                      sector_crop_list[15], sector_crop_list[15], sector_crop_list[15], 
                      sector_crop_list[14], sector_crop_list[14], sector_crop_list[14], 
                      sector_crop_list[14], sector_crop_list[14], None, None, None, 
                      sector_crop_list[15], sector_crop_list[15] ],
    
                    [ # threat zone 17
                      None, None, None, None, None, None, None, None,
                      sector_crop_list[4], sector_crop_list[4], sector_crop_list[4], 
                      sector_crop_list[4], sector_crop_list[4], sector_crop_list[4], 
                      sector_crop_list[4], sector_crop_list[4] ] ]


header = rdh.read_header(APS_FILE_NAME)
"""
for data_item in sorted(header):
    print ('{} -> {}'.format(data_item, header[data_item]))

"""

data = rdh.read_data(APS_FILE_NAME)

df = zpd.get_hit_rate_stats(THREAT_LABELS)

def chart_hit_rate_stats(df_summary):
    #fig, ax = plt.subplots(figsize=(15,5))
    sns.barplot(ax=ax, x=df_summary['Zone'], y=df_summary['pct']*100)

chart_hit_rate_stats(df)

zpd.print_hit_rate_stats(df)

print(zpd.get_subject_labels(THREAT_LABELS, '00360f79fd6e02781457eda48f85da90'))
label = zpd.get_subject_zone_label(13, zpd.get_subject_labels(THREAT_LABELS, '00360f79fd6e02781457eda48f85da90'))
print(np.array(label))
vimg.plot_image_set(APS_FILE_NAME)
an_img = vimg.get_single_image(APS_FILE_NAME, 0)

fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))

axarr[0].imshow(an_img, cmap=COLORMAP)
plt.subplot(122)
plt.hist(an_img.flatten(), bins=256, color='c')
plt.xlabel("Raw Scan Pixel Value")
plt.ylabel("Frequency")
plt.show()

img_rescaled = vimg.convert_to_grayscale(an_img)

fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))

axarr[0].imshow(img_rescaled, cmap=COLORMAP)
plt.subplot(122)
plt.hist(img_rescaled.flatten(), bins=256, color='c')
plt.xlabel("Grayscale Pixel Value")
plt.ylabel("Frequency")
plt.show()

img_high_contrast = vimg.spread_spectrum(img_rescaled)

fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))

axarr[0].imshow(img_high_contrast, cmap=COLORMAP)
plt.subplot(122)
plt.hist(img_high_contrast.flatten(), bins=256, color='c')
plt.xlabel("Grayscale Pixel Value")
plt.ylabel("Frequency")
plt.show()

fig, axarr = plt.subplots(nrows=4, ncols=4, figsize=(10,10))
    
i = 0
for row in range(4):
   for col in range(4):
        an_img = vimg.get_single_image(APS_FILE_NAME, i)
        img_rescaled = vimg.convert_to_grayscale(an_img)
        img_high_contrast = vimg.spread_spectrum(img_rescaled)
        if zone_slice_list[0][i] is not None:
            masked_img = vimg.roi(img_high_contrast, zone_slice_list[0][i])
            resized_img = cv2.resize(masked_img, (0,0), fx=0.1, fy=0.1)
            axarr[row, col].imshow(resized_img, cmap=COLORMAP)
        i += 1

fig, axarr = plt.subplots(nrows=4, ncols=4, figsize=(10,10))
    
i = 0
for row in range(4):
    for col in range(4):
        an_img = vimg.get_single_image(APS_FILE_NAME, i)
        img_rescaled = vimg.convert_to_grayscale(an_img)
        img_high_contrast = vimg.spread_spectrum(img_rescaled)
        if zone_slice_list[0][i] is not None:
            masked_img = vimg.roi(img_high_contrast, zone_slice_list[0][i])
            cropped_img = vimg.crop(masked_img, zone_crop_list[0][i])
            resized_img = cv2.resize(cropped_img, (0,0), fx=0.1, fy=0.1)
            axarr[row, col].imshow(resized_img, cmap=COLORMAP)
        i += 1
        
an_img = vimg.get_single_image(APS_FILE_NAME, 0)
img_rescaled = vimg.convert_to_grayscale(an_img)
img_high_contrast = vimg.spread_spectrum(img_rescaled)
masked_img = vimg.roi(img_high_contrast, zone_slice_list[0][0])
cropped_img = vimg.crop(masked_img, zone_crop_list[0][0])
normalized_img = vimg.normalize(cropped_img)
print ('Normalized: length:width -> {:d}:{:d}|mean={:f}'.format(len(normalized_img), len(normalized_img[0]), normalized_img.mean()))
print (' -> type ', type(normalized_img))
print (' -> shape', normalized_img.shape)

an_img = vimg.get_single_image(APS_FILE_NAME, 0)
img_rescaled = vimg.convert_to_grayscale(an_img)
img_high_contrast = vimg.spread_spectrum(img_rescaled)
masked_img = vimg.roi(img_high_contrast, zone_slice_list[0][0])
cropped_img = vimg.crop(masked_img, zone_crop_list[0][0])
normalized_img = vimg.normalize(cropped_img)
zero_centered_img = vimg.zero_center(normalized_img)
print ('Zero Centered: length:width -> {:d}:{:d}|mean={:f}'.format(len(zero_centered_img), len(zero_centered_img[0]), zero_centered_img.mean()))
print ('Conformed: Type ->', type(zero_centered_img), 'Shape ->', zero_centered_img.shape)