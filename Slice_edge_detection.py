"""
    ECE 6258: Digital Image Processing
    AND Project Code - Edge Detection

    Mighten Yip
    Mercedes Gonzalez
"""

from os.path import join, isfile
from os import listdir
import os
import dippykit as dip
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

# ____ INITIALIZATION  ____________________________________________________
# Set path of original image that was image enhanced
root_path = "C:/Users/might/Desktop/Neurons_raw/edge/original"
file_type = ".png"
file_list = [f for f in listdir(root_path) if isfile(join(root_path, f)) & f.endswith(file_type)]
filename = file_list[0]
original = cv2.imread(join(root_path, filename), cv2.IMREAD_GRAYSCALE)

# Set path of enhanced images to have cell detection
root_path = "C:/Users/might/Desktop/Neurons_raw/edge/canny"
save_path = "C:/Users/might/Desktop/Neurons_raw/edge/detect/"
file_list = [f for f in listdir(root_path) if isfile(join(root_path, f)) & f.endswith(file_type)]

edge_name = ['Original_detect', 'contrastStretch_detect', 'intensitySlice_detect', 'histEq_detect',
             'AdaptGauss_histEq_detect', 'AdaptMed_histEq_detect', 'AdaptBilat_histEq_detect',
             'unsharpMed_detect', 'unsharpBilat_detect']
# print(original.shape[1])
num_detect = np.zeros(9)
for count, filename in enumerate(file_list):
    print(count)
    filename = file_list[count]
    img = cv2.imread(join(root_path,filename),cv2.IMREAD_GRAYSCALE)
    cell_count = []
    print("Time to cell detect")

    # kernel = np.ones((5,5),np.uint8)
    # dilation = cv2.dilate(img,kernel,iterations=1) # Check to see how dilating the edges may look
    # closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel) # Check to see how closing may help find cells
    # # Perform the Hough Circle Transform
    cells = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT,1.0,100,None,80,28,10,50) # img.shape[1]/8: used for min dist b/w cells
    print("Sorting...")
    if cells is not None:
        cells = np.round(cells[0,:]).astype("uint8")
        cells[:, 0] += 640
        cells[:, 1] += 512
        # print(cells[:,0])
        for (x,y,r) in cells:
            print("Begin cell drawing")
            detect = cv2.circle(img,(x,y),r,(255,0,0),2)
            cell_count.append(r)
        edge_title = edge_name[count] + '.png'
        save_imageName = os.path.join(save_path, os.path.basename(edge_title))
        plt.imshow(detect, 'gray') # Use detect, dilation, or closing
        plt.show()
        plt.savefig(save_imageName)
        plt.close()
        print("Finished cell detection: cell count= ",len(cell_count))
        num_detect[count] = len(cell_count)


