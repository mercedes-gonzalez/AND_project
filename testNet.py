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

def detect_img(yolo,imgPath):
    image = Image.open(imgPath)
    r_image = yolo.detect_image(image=image)
    # r_image.show()
    return r_image

root_path = 'C:/Users/mgonzalez91/Dropbox (GaTech)/Coursework/SU20 - Digital Image Processing/AND_Project/testing_images/histEq_cropped_aug/'
save_path = 'C:/Users/mgonzalez91/Dropbox (GaTech)/Coursework/SU20 - Digital Image Processing/AND_Project/testing_images/histEq_cropped_aug/annotated'

file_list = [f for f in listdir(root_path) if isfile(join(root_path, f)) & f.endswith(".png")]
my_yolo = YOLO() # start yolo session

for f in file_list:
    annotated = detect_img(my_yolo,imgPath=join(root_path,f))
    annotated.save(join(save_path,f),"PNG")

my_yolo.close_session() # end yolo session
