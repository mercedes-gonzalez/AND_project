from data_aug.data_aug import *
from data_aug.bbox_util import *
import cv2 
import pickle as pkl
import numpy as np 
import matplotlib.pyplot as plt
from libs.pascal_voc_io import PascalVocReader
from libs.pascal_voc_io import XML_EXT


def loadPascalXMLByFilename(xmlPath):
    if xmlPath is None:
        return
    if os.path.isfile(xmlPath) is False:
        return

    # self.set_format(FORMAT_PASCALVOC)

    tVocParseReader = PascalVocReader(xmlPath)
    shapes = tVocParseReader.getShapes()
    return shapes
    
def resizeAndCrop(raw,newSize):
    xsize,ysize = raw.shape
    minDimension = min((x_raw,y_raw))
    xmin = math.floor(xsize/2-minDimension/2)
    ymin = math.floor(ysize/2-minDimension/2)

    width = minDimension
    height = minDimension

    cropped = raw[xmin:xmin+height,ymin:ymin+width]
    resized = cv2.resize(cropped,(newSize,newSize),interpolation=INTER_NEAREST)
    return resized

def resizeBoxes(bboxes,newSize,origSize,minDimension):
    xsize = origSize[0]
    ysize = origSize[1]
    newBoxes = np.zeros((bboxes.shape))
    
    # Transform x's
    newBoxes[:,0] = newSize*((bboxes[:,0]+(xsize/2))-(xsize-minDimension)/2)/minDimension 
    newBoxes[:,2] = newSize*((bboxes[:,2]+(xsize/2))-(xsize-minDimension)/2)/minDimension 
    
    # Transform y's
    newBoxes[:,1] = newSize*((bboxes[:,1]+(ysize/2))-(ysize-minDimension)/2)/minDimension 
    newBoxes[:,3] = newSize*((bboxes[:,3]+(ysize/2))-(ysize-minDimension)/2)/minDimension 

    return newboxes

img = cv2.imread("C:/Users/mgonzalez91/Dropbox (GaTech)/Coursework/SU20 - Digital Image Processing/AND_Project/slice_images_raw/subset_images/slice_3-14-2018_1.tiff")[:,:,::-1] #OpenCV uses BGR channels
bboxes = loadPascalXMLByFilename("C:/Users/mgonzalez91/Dropbox (GaTech)/Coursework/SU20 - Digital Image Processing/AND_Project/slice_images_raw/subset_images/slice_3-14-2018_1.xml")
print('bboxes = ',bboxes)

newSize = 640 # side length of a square input image
resizedI = resizeAndCrop(I,newSize)
newBoxes = resizeBoxes(bboxes,newSize,origSize,minDimension)
print('newboxes = ',newBoxes)
plt.imshow(draw_rect(resizedI,newBoxes))
plt.show()