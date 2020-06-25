"""
    ECE 6258: Digital Image Processing
    AND Project Code

    Mighten Yip
    Mercedes Gonzalez
"""

from os.path import join, isfile
from os import listdir
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
# ____ INITIALIZATION  ____________________________________________________
# Set path where all the images are, get list of all tiff files in that dir
root_path = "C:/Users/mgonzalez91/Dropbox (GaTech)/Coursework/SU20 - Digital Image Processing/AND_Project/slice_images_raw/subset_images/"
file_type = ".tiff"
file_list = [f for f in listdir(root_path) if isfile(join(root_path, f)) & f.endswith(file_type)]

filename = file_list[0]
img = cv2.imread(join(root_path,filename),cv2.IMREAD_GRAYSCALE)
imgsize = img.shape
M = imgsize[0]
N = imgsize[1]
# _________________________________________________________________________

# --- Various Filtering  ---------------------------------------------------------------------------
diam = 3 # Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace .
sigma_color = 1   # Filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel 
                    # neighborhood (see sigmaSpace ) will be mixed together, resulting in larger areas of semi-equal color
sigma_space = 100   # Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence 
                    # each other as long as their colors are close enough (see sigmaColor ). When d>0 , it specifies the neighborhood
                    #  size regardless of sigmaSpace . Otherwise, d is proportional to sigmaSpace 

gaussian_blur = cv2.GaussianBlur(img,(9,9),0)
median_blur = cv2.medianBlur(img,3)
bilateral_blur = cv2.bilateralFilter(img, diam, sigma_color, sigma_space)

# --- Unsharp Algorithm  ---------------------------------------------------------------------------
alpha = 2
beta = -.5
gamma =1

gaussian_unsharp = cv2.addWeighted(gaussian_blur,alpha,img,beta,gamma)
median_unsharp = cv2.addWeighted(median_blur,alpha,img,beta,gamma)
bilateral_unsharp = cv2.addWeighted(bilateral_blur,alpha,img,beta,gamma)

# --- Masking / Thresholding ---------------------------------------------------------------------------
threshold_perc = .9 # pixel intensity threshold as percentage (quantiles)
max_percent = .99 # set max value as this % of the max intensity 
blocksize = 151 # bigger blocksize masks bigger blobs
C = 13 # bigger C 

gaussian_mask = cv2.adaptiveThreshold(gaussian_unsharp,max_percent*np.max(gaussian_blur),cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,blocksize,C)
median_mask = cv2.adaptiveThreshold(median_blur,max_percent*np.max(gaussian_blur),cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,blocksize,C)
bilateral_mask = cv2.adaptiveThreshold(bilateral_blur,max_percent*np.max(gaussian_blur),cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,blocksize,C)

# --- Laplacian  ---------------------------------------------------------------------------
ddepth = cv2.CV_32F
ker = 111
gaussian_laplace = cv2.Laplacian(gaussian_blur,ddepth, ker)
median_laplace = cv2.Laplacian(median_blur,ddepth, ker)
bilateral_laplace = cv2.Laplacian(bilateral_blur,ddepth, ker)

# --- Contouring  ---------------------------------------------------------------------------
gaussian_Contour, _ = cv2.findContours(gaussian_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
median_contour, _ = cv2.findContours(median_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
bilateral_contour, _ = cv2.findContours(bilateral_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

plotFilter = 2 # 0=gauss, 1=median, 2=bilateral
if plotFilter == 0:
    filtername = "Gaussian"
    filtered = gaussian_blur
    sharpened = gaussian_unsharp
    masked = gaussian_mask
    laplacian = gaussian_laplace
    contourlist = gaussian_Contour
elif plotFilter == 1:
    filtername = "Median"
    filtered = median_blur
    sharpened = median_unsharp
    masked = median_mask
    laplacian = median_laplace
    contourlist = median_contour
elif plotFilter == 2:
    filtername = "Bilateral"
    filtered = bilateral_blur
    sharpened = bilateral_unsharp
    masked = bilateral_mask
    laplacian = bilateral_laplace
    contourlist = bilateral_contour

contour = np.copy(img) # will draw contours over original img

print('Num contours = ',len(contourlist)) # debug
for cnt in contourlist:
    # measure minimum enclosing circle for each contour detected
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    if True: # debuglol
    # if cv2.contourArea(cnt) < .1*M*N and cv2.contourArea(cnt) > .05*M*N and cv2.contourArea(cnt) > (radius*radius*3.1415*.5):
        cv2.drawContours(contour, cnt, -1, (1,1,1), 5)

        # M = cv2.moments(cnt)
        # cx = int(M['m10']/M['m00'])
        # cy = int(M['m01']/M['m00'])
        
        # centroids.append([cx, cy])
        # cv2.drawMarker(contour,(cx,cy),(color_num, color_num, color_num), cv2.MARKER_CROSS,10,1)

# ___ PLOTTING ____________________________________________________________________________
img_array = [ img, filtered, sharpened,contour, masked, laplacian]  
title_array = [ 'Original', 'Filtered', 'Sharpened', 'Contour','Filtered Masked', 'Laplacian']

fig, axs = plt.subplots(2,3)
fig.suptitle(filtername, fontsize=20)
axs = axs.flatten()
fig.subplots_adjust(wspace=0,hspace=.1)
for i,ax in enumerate(axs):
    title_str = title_array[i]
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title_str)
    ax.imshow(img_array[i],cmap='gray')
plt.show()

# IGNORE ____________________________________________________________________________________________

# img_array = [ gaussian_blur, median_blur, bilateral_blur, gaussian_mask, median_mask, bilateral_mask ]
# title_array = [ 'Gaussian Filter', 'Median Filter', 'Bilateral Filter', 'Gaussian Masked','Median Masked','Bilateral Masked']

# fig, axs = plt.subplots(2,3)
# axs = axs.flatten()
# for i,ax in enumerate(axs):
#     title_str = title_array[i]
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_title(title_str)
#     ax.imshow(img_array[i],cmap='gray')
# plt.show()

# Fourier Transform of Original image
    # img_fft = dip.fft2(img)
    # img_fft = dip.fftshift(img_fft)
    # img_fft = np.log(np.abs(img_fft))
    # plt.imshow(img_fft)
    # dip.show()