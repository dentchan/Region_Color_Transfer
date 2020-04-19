## Author : Damian Holmes

import numpy as np

from PIL import Image
import scipy.spatial as sp
import cv2
import matplotlib.pyplot as plt




##load image
input_image = cv2.imread("flowers.jpg")

toChangeColors = input_image.copy()

##convert input image to HSV color space, as to detect red color regions
hsv_image=cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

## Generate lower mask (0-5) and upper mask (175-180) of RED
## This part can be alter with needs
## Google inRange hsv color space to understand more
mask1 = cv2.inRange(hsv_image, (0,50,20), (5,255,255))
mask2 = cv2.inRange(hsv_image, (175,50,20), (180,255,255))

## Merge the mask and crop the red regions
mask = cv2.bitwise_or(mask1, mask2)

## Crop the  detected red regions out from the input image
region_croped = cv2.bitwise_and(input_image, input_image, mask=mask)

## Load reference image
reference_image=cv2.imread('red.jpg')

## Convert croped red regions to LAB color space
region_croped = cv2.cvtColor(region_croped, cv2.COLOR_BGR2LAB)

## Convert reference image to LAB color space
reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2LAB)

## Apply Mean Standard Deviation(LAB color space) color transfer
## You can also skip the LAB conversion part to obtain a different kind of result
## Code obtained from https://github.com/pengbo-learn/python-color-transfer [reference 1]

## Start of color transfer applying
mean_in = np.mean(region_croped, axis=(0, 1), keepdims=True)
mean_ref = np.mean(reference_image, axis=(0, 1), keepdims=True)
std_in = np.std(region_croped, axis=(0, 1), keepdims=True)
std_ref = np.std(reference_image, axis=(0, 1), keepdims=True)
color_transfered_region = (region_croped - mean_in) / std_in * std_ref + mean_ref
color_transfered_region[color_transfered_region < 0] = 0
color_transfered_region[color_transfered_region > 255] = 255
color_transfered_region=color_transfered_region.astype('uint8')
## End of color transfer applying

## Convert color transfered region back to BGR format
color_transfered_region = cv2.cvtColor(color_transfered_region, cv2.COLOR_LAB2BGR)

## mask3 is the inverted mask obtained above from detecting red regions
## red regions in mask 3 is now black out
mask3 = cv2.bitwise_not(mask)

## res1 applied mask3 on to input image
## res1 is now input image but red regions are cropped away
res1 = cv2.bitwise_and(input_image,input_image,mask=mask3)

## res2 applied mask (obtained from merging 2 red regions detecting mask)
## res2 is now red regions cropped from input image, but applied color transfer
res2 = cv2.bitwise_and(color_transfered_region, color_transfered_region, mask = mask)

## final output are res1 and res2 overlapping to achieve region color transfer
## red regions detecting, cropping and overlapping are ideas obtained from (reference 2)
## https://www.learnopencv.com/invisibility-cloak-using-color-detection-and-segmentation-with-opencv/
final_output = cv2.addWeighted(res1,1,res2,1,0)


## Below are just resizing and showing final output
width = 635
height = 635
dim = (width, height)
final_output = cv2.resize(final_output, dim, interpolation = cv2.INTER_LINEAR)
input_image = cv2.resize(input_image, dim, interpolation = cv2.INTER_LINEAR)

numpy_horizontal_concat = np.concatenate((input_image, final_output), axis=1)


cv2.imshow("Result", numpy_horizontal_concat)

cv2.waitKey()
