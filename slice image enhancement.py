# ECE 6258 Slice image enhancement
# Mighten Yip & Mercedes Gonzalez
import dippykit as dip
import numpy as np
import cv2
import matplotlib.pyplot as plt

I = dip.imread(
    "C:/Users/might/Dropbox (GaTech)/Shared folders/AND_Project/slice_images_raw/subset_images/slice_3-14-2018_1.tiff")
arrayI = np.asarray(I)
I = dip.im_to_float(I)
M, N = np.shape(I)
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
# To the eye, there is not much difference in the images, quantitatively, there is minimal difference.
# MSE = 0.0014015751637218763
# print(np.sum(((csImg/255)-(C/255))**2)/(M*N))
# plt.figure(3)
# plt.hist(C.flatten(),256,[0,256],color='b')
# plt.hist(csImg.flatten(),256,[0,256],color='r')

# Idea 4: Intensity-level slicing
D = I*255
# Create an zeros array to store the sliced image
isImg = np.zeros((M,N), dtype='uint8')

# Specify the min and max range
min_range = 141
max_range = 146

# Loop over the input image and if pixel value lies in desired range set it to 255 otherwise set it to 0.
for i in range(M):
    for j in range(N):
        if D[i,j] > min_range and D[i,j] < max_range:
            isImg[i,j] = 255
        else:
            isImg[i,j] = 0#D[i,j] # 'leave everything' or 0 if 'remove everything'

# Idea 5: Adaptive thresholding (matlab has an adaptive thresholding thing...could show version of it?

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

# plt.figure(1)
# plt.subplot(221)
# plt.plot(cdf, color = 'b')
# plt.subplot(222)
# plt.hist(E.flatten(),256,[0,256], color = 'r')
# plt.xlim([0,256])
# plt.legend(('cdf','histogram'), loc = 'upper left')
# plt.subplot(223)
# plt.plot(cdf_heq, color='b')
# plt.title('Histogram equalization')
# plt.subplot(224)
# plt.hist(histEq.flatten(),256,[0,256], color='r')
# plt.xlim([0,256])
# plt.legend(('cdf','histogram'), loc = 'upper left')
# plt.show()

# Plot all seg/filter ideas together
plt.figure(2)
plt.subplot(231)
plt.imshow(I,'gray')
plt.title('Original image')
# plt.subplot(232)
# plt.imshow(A/255,'gray')
# plt.title('Binarize thresholding')
# plt.subplot(233)
# plt.title('Filtering')
plt.subplot(234)
plt.imshow(csImg/255, 'gray')
plt.title('Contrast stretching')
plt.subplot(235)
plt.imshow(isImg/255,'gray')
plt.title('Intensity-level slicing')
# plt.subplot(236)
# plt.imshow(histEqImg,'gray')
# plt.title('Histogram Equalization')
plt.show()

