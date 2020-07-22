"""
    ECE 6258: Digital Image Processing
    AND Project Code

    Mighten Yip
    Mercedes Gonzalez

"""
import os
from os.path import join, isfile
from os import listdir
import dippykit as dip
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from PIL import Image
from xml.etree import ElementTree as et

def changeXML(xmlPath,name,root_path):
    if xmlPath is None:
        return
    if os.path.isfile(xmlPath) is False:
        return
    tree = et.parse(xmlPath)
    tree.find('.//filename').text = name + '.png'
    tree.find('.//path').text = root_path + name + '.png'
    tree.write(xmlPath)


# ____ INITIALIZATION  ____________________________________________________
# Set path where all the images are, get list of all tiff files in that dir
root_path = "C:/Users/mgonzalez91/Dropbox (GaTech)/Coursework/SU20 - Digital Image Processing/AND_Project/testing_images/untouched/"
save_path = "C:/Users/mgonzalez91/Dropbox (GaTech)/Coursework/SU20 - Digital Image Processing/AND_Project/testing_images/histEq/"
file_type = ".png"
file_list = [f for f in listdir(root_path) if isfile(join(root_path, f)) & f.endswith(file_type)]
file_list_xml = [f for f in listdir(save_path) if isfile(join(save_path, f)) & f.endswith('.xml')]
dim = (416,416)

for count,filename in enumerate(file_list):
    I = cv2.imread(join(root_path,filename),cv2.IMREAD_GRAYSCALE)

    # hist eq
    E = np.asarray(I)
    flat = E.flatten()
    hist, bins = np.histogram(flat,256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    cdf_num = (cdf - cdf.min()) * 255
    cdf_den = cdf.max() - cdf.min()
    cdf_heq = cdf_num/cdf_den
    cdf_heq = cdf_heq.astype('uint8')
    histEq = cdf_heq[flat]
    hist2, bins2 = np.histogram(histEq,256,[0,256])
    cdf_norm_heq = cdf_heq * hist2.max()/cdf_heq.max()
    histEqImg = np.reshape(histEq,I.shape)

    equalizedImage = Image.fromarray(histEqImg)
    save_filename = join(save_path,filename)
    equalizedImage.save(save_filename)
    # break
    # plt.subplot(1,2,1)
    # plt.imshow(filt,cmap='gray')
    # plt.subplot(1,2,2)
    # plt.imshow(histEqImg,cmap='gray')
    # plt.show()

    # break

# for count,filename in enumerate(file_list_xml):
#     # new xml
#     base,ext = os.path.splitext(filename)
#     fullfile = join(save_path,filename)
#     changeXML(fullfile,base,save_path)
        
#     print(base)

    # break
