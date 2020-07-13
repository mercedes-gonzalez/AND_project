from data_aug.data_aug import *
from data_aug.bbox_util import *
import cv2 
import pickle as pkl
import numpy as np 
import matplotlib.pyplot as plt
from libs.pascal_voc_io import PascalVocReader
from libs.pascal_voc_io import XML_EXT
import math

def loadPascalXMLByFilename(xmlPath):
    if xmlPath is None:
        return
    if os.path.isfile(xmlPath) is False:
        return
    tVocParseReader = PascalVocReader(xmlPath)
    shapes = tVocParseReader.getShapes()
    return shapes
    
def resizeAndCrop(raw,newSize,bboxes):
    xsize,ysize,_ = raw.shape
    minDimension = min((xsize,ysize))
    width = minDimension
    height = minDimension

    left = np.zeros((len(bboxes),5))
    right = np.zeros((len(bboxes),5))
    old = np.zeros((len(bboxes),5))
    for i,box in enumerate(bboxes):
        # Reformat data for augmentation
        coords = box[1]
        x1 = coords[0][0]
        y1 = coords[0][1]
        x2 = coords[1][0]
        y2 = coords[2][1]
        old[i,:] = np.array([x1, y1, x2, y2, 0])

    # CROP LEFT -------------------------------------------
    cropped_L = raw[0:height,0:width].copy()
    resized_L = cv2.resize(cropped_L,(newSize,newSize),interpolation=cv2.INTER_NEAREST)

    # Transform bounding boxes ------
    
    # Transform x's
    left[:,0] = (newSize/minDimension)*old[:,0]
    left[:,2] = (newSize/minDimension)*old[:,2]
    
    # Transform y's
    left[:,1] = (newSize/minDimension)*old[:,1]
    left[:,3] = (newSize/minDimension)*old[:,3]

    # CROP RIGHT -------------------------------------------
    xmin = xsize - minDimension
    ymin = ysize - minDimension
    cropped_R = raw[xmin:,ymin:].copy()
    resized_R = cv2.resize(cropped_R,(newSize,newSize),interpolation=cv2.INTER_NEAREST)

    # Transform bounding boxes ------
    # Transform x's
    right[:,0] = (newSize/minDimension)*(old[:,0]-ymin)
    right[:,2] = (newSize/minDimension)*(old[:,2]-ymin)
    
    # Transform y's
    right[:,1] = (newSize/minDimension)*(old[:,1])
    right[:,3] = (newSize/minDimension)*(old[:,3])

    return resized_R, resized_L, left, right, old

I = cv2.imread("C:/Users/mgonzalez91/Dropbox (GaTech)/Coursework/SU20 - Digital Image Processing/AND_Project/slice_images_raw/subset_images/slice_3-14-2018_2.tiff")[:,:,::-1] #OpenCV uses BGR channels
bboxes = loadPascalXMLByFilename("C:/Users/mgonzalez91/Dropbox (GaTech)/Coursework/SU20 - Digital Image Processing/AND_Project/slice_images_raw/subset_images/slice_3-14-2018_2.xml")
# print('bboxes = ',bboxes)

newSize = 640 # side length of a square input image
resized_R,resized_L,left,right,old = resizeAndCrop(raw=I,newSize=newSize,bboxes=bboxes)

fig,(ax1,ax2,ax3) = plt.subplots(1,3)
ax1.imshow(draw_rect(I,old))
ax2.imshow(draw_rect(resized_L,left))
ax3.imshow(draw_rect(resized_R,right))
plt.show()