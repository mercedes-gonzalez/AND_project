from data_aug.data_aug import *
from data_aug.bbox_util import *
import cv2 
import pickle as pkl
import numpy as np 
import matplotlib.pyplot as plt
from libs.pascal_voc_io import PascalVocReader,PascalVocWriter
from libs.pascal_voc_io import XML_EXT
import math
from PIL import Image
from os.path import join, isfile
from os import listdir

# FUNCTION DEFINITIONS _____________________________________________
def loadPascalXMLByFilename(xmlPath):
    if xmlPath is None:
        return
    if os.path.isfile(xmlPath) is False:
        return
    tVocParseReader = PascalVocReader(xmlPath)
    shapes = tVocParseReader.getShapes()
    return shapes
    
def resizeAndCrop(raw,newSize,bboxes):
    xsize,ysize = raw.shape
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

def saveXML(img,filename, shapes, imagePath, lineColor=None, fillColor=None, databaseSrc=None):
    imgFolderPath = os.path.dirname(imagePath)
    imgFolderName = os.path.split(imgFolderPath)[-1]
    imgFileName = os.path.basename(imagePath)
    #imgFileNameWithoutExt = os.path.splitext(imgFileName)[0]
    # Read from file path because self.imageData might be empty if saving to
    # Pascal format
    imgShape = img.shape
    writer = PascalVocWriter(foldername=imgFolderName, filename=imgFileName, imgSize=imgShape, localImgPath=imagePath)

    # if shapes.size == 5:
    #     writer.addBndBox(int(shapes[0]), int(shapes[1]), int(shapes[2]), int(shapes[3]), 'neuron', False)
    # else:
    for shape in shapes:
        writer.addBndBox(int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3]), 'neuron', False)

    writer.save(targetFile=filename)
    # print('Successfully saved.\n')
    return

def histEqual(I):
    # Preprocess left cropped images
    E = np.asarray(I)
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

    return histEqImg

# Set path where all the images are, get list of all tiff files in that dir
lab_comp =True
if lab_comp == True:
    root_path = "C:/Users/myip7/AND_Project_MG/preprocessed_training_data/uncropped/"
    save_path = "C:/Users/myip7/AND_Project_MG/preprocessed_training_data/cropped_fixed_filename/"
else:
    root_path = "C:/Users/mgonzalez91/Dropbox (GaTech)/Coursework/SU20 - Digital Image Processing/AND_Project/slice_images_raw/subset_images/"
    save_path = "C:/Users/mgonzalez91/Dropbox (GaTech)/Coursework/SU20 - Digital Image Processing/AND_Project/slice_images_raw/subset_images/"
file_type = ".tiff"
file_list = [f for f in listdir(root_path) if isfile(join(root_path, f)) & f.endswith(file_type)]
newSize = 640 # side length of a square input image, 20x32

# CROPPING
for count, filename in enumerate(file_list):
    # fullname = os.path.join(save_path,filename)
    base = os.path.basename(filename)
    fileID = os.path.splitext(base)[0]
    xmlname = root_path + fileID + '.xml'

    # Read file and load xml data
    I = cv2.imread(join(root_path,filename),cv2.IMREAD_GRAYSCALE)
    bboxes = loadPascalXMLByFilename(join(root_path,xmlname))

    # Crop to square, resize to desired dims
    resized_R,resized_L,leftbox,rightbox,oldbox = resizeAndCrop(raw=I,newSize=newSize,bboxes=bboxes)

    # Transform right cropped images again because there will be duplicate cells
    Rtransforms = Rotate(angle=90)


    # only save image and xml if bboxes exist in the image
    if rightbox.size != 0:
        RImg, RFormat = Rtransforms(resized_R, rightbox)
        # be sure no bboxes are outside the cropped image of right cropped
        delete_index_R = np.zeros((RFormat.shape)).astype('bool')
        for count, row in enumerate(RFormat):
            if  not(0 <= row[0] <= newSize) or not(0 <= row[2] <= newSize):
                delete_index_R[count,:] = 1
            if  not(0 <= row[1] <= newSize) or not(0 <= row[3] <= newSize):
                delete_index_R[count,:] = 1
        NR = RFormat[~delete_index_R].size
        rightfinal = RFormat[~delete_index_R].reshape((int(NR/5),5))
        RFilename = save_path + fileID + '_R.tiff'
        RImage = Image.fromarray(RImg)
        RImage.save(RFilename)
        RXmlname = save_path + fileID + '_R.xml'
        saveXML(img=RImg,filename=RXmlname, shapes=rightfinal, imagePath=RFilename)
    
    if leftbox.size != 0:
        # be sure no bboxes are outside the cropped image of left cropped
        delete_index_L = np.zeros((leftbox.shape)).astype('bool')
        for count, row in enumerate(leftbox):
            if  not(0 <= row[0] <= newSize) or not(0 <= row[2] <= newSize):
                delete_index_L[count,:] = 1
            if  not(0 <= row[1] <= newSize) or not(0 <= row[3] <= newSize):
                delete_index_L[count,:] = 1
        NL = leftbox[~delete_index_L].size
        leftfinal = leftbox[~delete_index_L].reshape((int(NL/5),5))
        LFilename = save_path + fileID + '_L.tiff'
        LImage = Image.fromarray(resized_L)
        LImage.save(LFilename)
        LXmlname = save_path + fileID + '_L.xml'
        saveXML(img=resized_L,filename=LXmlname, shapes=leftfinal, imagePath=LFilename)
        print('\n',count)
    # Show resizedcr/cropped images
    # fig,(ax1,ax2,ax3) = plt.subplots(1,3)
    # ax1.imshow(draw_rect(I,oldbox),cmap='gray')
    # ax2.imshow(draw_rect(resized_L,leftbox),cmap='gray')
    # ax3.imshow(draw_rect(RImg,RFormat),cmap='gray')
    # plt.show()

    # break # use this for checking


