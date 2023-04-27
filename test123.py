'''
import cv2
img = cv2.imread('aaa.jpg')
img = cv2.resize(img, (300, 300))
#lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#a_component = lab[:,:,1]
th = cv2.threshold(img,140,255,cv2.THRESH_BINARY)[1]
#cv2.imshow('frame', th)
blur = cv2.GaussianBlur(th,(13,13), 11)

heatmap_img = cv2.applyColorMap(blur, cv2.COLORMAP_JET)

super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0)
#cv2.show(super_imposed_img)
cv2.imshow('frame', super_imposed_img)
cv2.waitKey(0)

'''
#---------------------
import numpy as np
from cv2 import cv2
from skimage import exposure
img = cv2.imread('aaa.jpg')
img = cv2.resize(img, (300, 300))
cv2.imshow('frame', img)
cv2.waitKey(0)

        #get color map
#cam = getMap(img)
map_img = exposure.rescale_intensity(img, out_range=(0, 255))
map_img = np.uint8(map_img)
heatmap_img = cv2.applyColorMap(map_img, cv2.COLORMAP_JET)

        #merge map and frame
fin = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0)

        #show result
cv2.imshow('frame', fin)
cv2.waitKey(0)