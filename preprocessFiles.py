"""
    ECE 6258: Digital Image Processing
    AND Project Code

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
from PIL import Image
# ____ INITIALIZATION  ____________________________________________________
# Set path where all the images are, get list of all tiff files in that dir
root_path = "C:/Users/mgonzalez91/Dropbox (GaTech)/Coursework/SU20 - Digital Image Processing/AND_Project/slice_images_raw/preprocessed_training_data/neuron/"
file_type = ".tiff"
file_list = [f for f in listdir(root_path) if isfile(join(root_path, f)) & f.endswith(file_type)]

for count,filename in enumerate(file_list):
    I = cv2.imread(join(root_path,filename),cv2.IMREAD_GRAYSCALE)
    arrayI = np.asarray(I)
    I = dip.im_to_float(I)

    E = arrayI
    E = np.asarray(E)
    flat = E.flatten()
    # Find Cumulative distributive function (cdf)
    hist, bins = np.histogram(flat,256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()

    cdf_num = (cdf - cdf.min()) * 255
    cdf_den = cdf.max() - cdf.min()
    # re-normalize the cdf
    cdf_heq = cdf_num/cdf_den
    cdf_heq = cdf_heq.astype('uint8')

    histEq = cdf_heq[flat]
    hist2, bins2 = np.histogram(histEq,256,[0,256])
    cdf_norm_heq = cdf_heq * hist2.max()/cdf_heq.max()
    histEqImg = np.reshape(histEq,I.shape)

    equalizedImage = Image.fromarray(histEqImg)
    equalizedImage.save(filename)
    print(count)
