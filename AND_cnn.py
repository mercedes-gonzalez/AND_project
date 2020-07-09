"""
	ECE 6258: Digital Image Processing
	AND Project Code
	CNN Architecture

	Mighten Yip
	Mercedes Gonzalez
"""

from os.path import join, isfile
from os import listdir
import dippykit as dip
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

# ____ INITIALIZATION  ____________________________________________________
# Set path where all the images are, get list of all tiff files in that dir
# root_path = "C:/Users/mgonzalez91/Dropbox (GaTech)/Coursework/SU20 - Digital Image Processing/AND_Project/slice_images_raw/subset_images/"
root_path = "C:/Users/might/Dropbox (GaTech)/Shared folders/AND_Project/slice_images_raw/subset_images"
file_type = ".tiff"
file_list = [f for f in listdir(root_path) if isfile(join(root_path, f)) & f.endswith(file_type)]

# --- Select an image ---
filename = file_list[0]
img = cv2.imread(join(root_path,filename),cv2.IMREAD_GRAYSCALE)
imgsize = img.shape
M = imgsize[0]
N = imgsize[1]

ann_img = np.zeros((30,30,3)).astype('uint8')
ann_img[ 3 , 4 ] = 1 # this would set the label of pixel 3,4 as 1

cv2.imwrite( "ann_1.png" ,ann_img )