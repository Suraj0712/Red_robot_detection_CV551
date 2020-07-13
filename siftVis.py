import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('armModel/front.png',0)    

img2 = cv2.imread('fullRobot/003.jpg',0) # testImage
img2 = cv2.resize(img2,(0,0),fx=.5,fy=.5)


img1 = cv2.GaussianBlur(img1,(1,1),0)

sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers=10, edgeThreshold=10)
#minHessian = 400
#sift = cv2.xfeatures2d_SURF.create(hessianThreshold=minHessian)
img = img2

kp1 = sift.detect(img,None)
img=cv2.drawKeypoints(img,kp1,img)

cv2.imshow("model features",img)
cv2.waitKey(0)