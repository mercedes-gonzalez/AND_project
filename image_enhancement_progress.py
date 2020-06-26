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
# ____ INITIALIZATION  ____________________________________________________
# Set path where all the images are, get list of all tiff files in that dir
# root_path = "C:/Users/mgonzalez91/Dropbox (GaTech)/Coursework/SU20 - Digital Image Processing/AND_Project/slice_images_raw/subset_images/"
root_path = "C:/Users/might/Dropbox (GaTech)/Shared folders/AND_Project/slice_images_raw/subset_images"
file_type = ".tiff"
file_list = [f for f in listdir(root_path) if isfile(join(root_path, f)) & f.endswith(file_type)]

filename = file_list[0]
img = cv2.imread(join(root_path,filename),cv2.IMREAD_GRAYSCALE)
imgsize = img.shape
M = imgsize[0]
N = imgsize[1]
# _________________________________________________________________________
# Merging code from "slice image enhancement.py"
I = img
arrayI = np.asarray(I)
I = dip.im_to_float(I)
# M, N = np.shape(I)
# Idea 1: Binarize thresholding
A = I*255
# Specify a threshold 0-255
threshold = 143
# make all pixels < threshold black
for m in range(0,M):
    for n in range(0,N):
        if A[m,n] > threshold:
            A[m,n] = 1
        else:
            A[m,n] = 0

# Idea 2: Filtering
B = I*255
# Mercedes filtering tingy

# Idea 3: Contrast stretching
C = I*255
contrast = dip.contrast(C)
# print(contrast)
cmax = np.max(C) # cmax = 212
cmin = np.min(C) # cmin = 51
# print(cmax,cmin)
# Function to contrast stretch
def contrastStretch(image):
    iI = image # image input
    minI = 51   # minimum intensity (input)
    maxI = 212  # maxmimum intensity (input)
    minO = 0    # minimum intensity (output)
    maxO = 255  # maxmimum intensity (output)
    iO = (iI - minI) * (((maxO - minO) / (maxI - minI)) + minO) # image output
    return iO
csImg = contrastStretch(C)
# print(np.max(csImg),np.min(csImg))
# print(csImg)

# Idea 4: Intensity-level slicing
D = I*255
# Create an zeros array to store the sliced image
isImg = np.zeros((M,N), dtype='uint8')

# Specify the min and max range
min_range = 141
max_range = 147

# Loop over the input image and if pixel value lies in desired range set it to 255 otherwise set it to 0.
for i in range(M):
    for j in range(N):
        if D[i,j] > min_range and D[i,j] < max_range:
            isImg[i,j] = 255
        else:
            isImg[i,j] = D[i,j] # 'leave everything' or 0 if 'remove everything'

# Idea 6: Histogram equalization
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

# Plot all image enhancement ideas together
# plt.figure(1)
# plt.subplot(231)
# plt.imshow(I,'gray')
# plt.title('Original image')
# plt.subplot(232)
# plt.imshow(A/255,'gray')
binarize = A/255
# plt.title('Binarize thresholding')
# plt.subplot(233)
# plt.title('Filtering')
# plt.subplot(234)
# plt.imshow(csImg/255, 'gray')
csImg = csImg/255
# plt.title('Contrast stretching')
# plt.subplot(235)
# plt.imshow(isImg/255,'gray')
isImg = isImg/255
# plt.title('Intensity-level slicing')
# plt.subplot(236)
# plt.imshow(histEqImg,'gray')
# plt.title('Histogram Equalization')
# plt.show()


# --- Various Filtering  ---------------------------------------------------------------------------
diam = 75 # Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace .
sigma_color = 75   # Filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel 
                    # neighborhood (see sigmaSpace ) will be mixed together, resulting in larger areas of semi-equal color
sigma_space = 75   # Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence 
                    # each other as long as their colors are close enough (see sigmaColor ). When d>0 , it specifies the neighborhood
                    #  size regardless of sigmaSpace . Otherwise, d is proportional to sigmaSpace 

gaussian_blur = cv2.GaussianBlur(img,(11,11),0)
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
max_percent = .90 # set max value as this % of the max intensity 
blocksize = 211 # bigger blocksize masks bigger blobs
C = 2 # bigger C 

gaussian_mask = cv2.adaptiveThreshold(gaussian_blur,max_percent*np.max(gaussian_blur),cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,blocksize,C)
median_mask = cv2.adaptiveThreshold(median_blur,max_percent*np.max(gaussian_blur),cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,blocksize,C)
bilateral_mask = cv2.adaptiveThreshold(bilateral_blur,max_percent*np.max(gaussian_blur),cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,blocksize,C)

# --- Laplacian  ---------------------------------------------------------------------------
ddepth = cv2.CV_32F
ker = 11
gaussian_laplace = cv2.Laplacian(gaussian_blur,ddepth, ker)
median_laplace = cv2.Laplacian(median_blur,ddepth, ker)
bilateral_laplace = cv2.Laplacian(bilateral_blur,ddepth, ker)

# --- Contouring  ---------------------------------------------------------------------------
gaussian_Contour, _ = cv2.findContours(gaussian_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
median_contour, _ = cv2.findContours(median_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
bilateral_contour, _ = cv2.findContours(bilateral_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

plotFilter = 0 # 0=gauss, 1=median, 2=bilateral
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
img_array = [ img, binarize, masked, csImg, isImg, histEqImg, sharpened, contour, laplacian]
title_array = [ 'Original', 'Binarize Threshold','Adaptive Threshold', 'Contrast Stretching', 'Intensity-level Slicing',
                'Histogram Equalization', 'Sharpened', 'Contour', 'Laplacian']

fig, axs = plt.subplots(3,3)
# fig.suptitle('Image enhancement techniques on a brain slice image', fontsize=16) # formerly, used 'filtername'. Now, titling the entire subplots
axs = axs.flatten()
for i,ax in enumerate(axs):
    title_str = title_array[i]
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title_str)
    ax.imshow(img_array[i],cmap='gray')
fig.tight_layout()
fig.subplots_adjust(wspace=0,hspace=.2,top=0.90)
plt.show()

# # Something else to consider... built in denoising algorithms. May help with masking. but I don't want to do it today.
# template=1
# search=3
# h=cv2.NORM_L2
# blocksize=191
# C=25
# plt.figure()
# denoised=cv2.fastNlMeansDenoising(img,template,search,h)
# maskeddenoise= cv2.adaptiveThreshold(denoised,max_percent*np.max(denoised),cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,blocksize,C)
# plt.imshow(maskeddenoise,cmap='gray')
# plt.show()

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