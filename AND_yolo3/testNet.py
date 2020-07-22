"""
    Script to test neural network without yolo_video.py

    make sure that paths in yolo.py to model, classes, anchors, 
    and font are all absolute paths on your machine

    reads images in from root_path and saves annotated pngs to save_path
"""
import sys
from yolo import YOLO, detect_video
from PIL import Image
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import os 

mighten = True

def detect_img(yolo, imgPath, logName, timeLog):
    image = Image.open(imgPath)
    r_image = yolo.detect_image(image=image,logPath=logName, timeLog=timeLog)
    # r_image.show()
    return r_image


if mighten == True:
    # untouched Input
    root_path = 'C:/Users/might/Dropbox (GaTech)/Shared folders/AND_Project/testing_images/untouched_input/'
    save_path = 'C:/Users/might/Dropbox (GaTech)/Shared folders/AND_Project/testing_images/untouched_input/histEqNet_uIm_results/'
    # histEq Input
    # root_path = 'C:/Users/might/Dropbox (GaTech)/Shared folders/AND_Project/testing_images/histEq_input/'
    # save_path = 'C:/Users/might/Dropbox (GaTech)/Shared folders/AND_Project/testing_images/histEq_input/histEqNet_hIm_results/'

else:
    root_path = 'C:/Users/mgonzalez91/Dropbox (GaTech)/Coursework/SU20 - Digital Image Processing/AND_Project/testing_images/histEq_input/'
    save_path = 'C:/Users/mgonzalez91/Dropbox (GaTech)/Coursework/SU20 - Digital Image Processing/AND_Project/testing_images/histEq_input/histEq_trained_net_results/'
    # root_path = 'C:/Users/mgonzalez91/Dropbox (GaTech)/Coursework/SU20 - Digital Image Processing/AND_Project/testing_images/untouched_input/'
    # save_path = 'C:/Users/mgonzalez91/Dropbox (GaTech)/Coursework/SU20 - Digital Image Processing/AND_Project/testing_images/untouched_input/histEq_trained_net_results/'

file_list = [f for f in listdir(root_path) if isfile(join(root_path, f)) & f.endswith(".png")]
my_yolo = YOLO() # start yolo session

for f in file_list:
    base = os.path.basename(f)
    logID = save_path + os.path.splitext(base)[0] + '.txt'
    logTime = save_path + 'inference_time.txt'
    annotated = detect_img(my_yolo,imgPath=join(root_path,f),logName=logID,timeLog=logTime)
    annotated.save(join(save_path,f),"png")

my_yolo.close_session() # end yolo session
