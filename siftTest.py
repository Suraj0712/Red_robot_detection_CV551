import numpy as np
import cv2
from matplotlib import pyplot as plt


def floodFill(bin_im):
	im_floodfill = bin_im.copy()	 
	# Mask used to flood filling.
	# Notice the size needs to be 2 pixels than the image.
	h, w = bin_im.shape[:2]
	mask = np.zeros((h+2, w+2), np.uint8)	 
	# Floodfill from point (0, 0)
	cv2.floodFill(im_floodfill, mask, (0,0), 255);	 
	# Invert floodfilled image
	im_floodfill_inv = cv2.bitwise_not(im_floodfill)	 
	# Combine the two images to get the foreground.
	im_out = bin_im | im_floodfill_inv
	return im_out

def removeBackground(imgColor,imgGray):
	# # Convert BGR to HSV
	# hsv = cv2.cvtColor(imgColor, cv2.COLOR_BGR2HSV)
	# # define range of red color in HSV
	# low_tresh = np.array([0,1,0],np.uint8)
	# up_thresh = np.array([100,255,255],np.uint8)
	# # Threshold the HSV image to get only red
	# mask = cv2.inRange(imgColor, low_tresh, up_thresh)

	
	# cv2.imshow("edges",mask)
	# cv2.waitKey(0)


	# mask = floodFill(mask)
	# cv2.imshow("filled",mask)
	# cv2.waitKey(0)
	# cv2.imshow("Masked",cv2.bitwise_and(imgColor,imgColor,mask=mask))
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	# # Bitwise-AND mask and original image
	res = imgGray#cv2.bitwise_and(imgGray,imgGray, mask= mask)

	return res

img1 = cv2.imread('bodyModel/back.png',1)    
img1Gray = cv2.imread('bodyModel/back.png',0)  
        # queryImage

img2 = cv2.imread('fullRobot/002.jpg',1) # testImage
img2 = cv2.resize(img2,(0,0),fx=.5,fy=.5)

img2Gray = cv2.imread('fullRobot/002.jpg',0) # testImage
img2Gray = cv2.resize(img2Gray,(0,0),fx=.5,fy=.5)

img1 = cv2.GaussianBlur(img1,(3,3),0)
img1Gray = cv2.GaussianBlur(img1Gray,(3,3),0)

img2 = cv2.GaussianBlur(img2,(3,3),0)
img2Gray = cv2.GaussianBlur(img2Gray,(3,3),0)



img1 = removeBackground(img1,img1Gray)
img2 = removeBackground(img2,img2Gray)

rows,cols = img2.shape

#90 degree rotation

# M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
# img2 = cv2.warpAffine(img2,M,(cols,rows))
# 
# Affine Transform
# pts1 = np.float32([[50,50],[200,50],[50,200]])
# pts2 = np.float32([[50,50],[150,50],[50,200]])
# M = cv2.getAffineTransform(pts1,pts2)
# img2 = cv2.warpAffine(img2,M,(cols,rows))


# Initiate SIFT detector
sift1 = cv2.xfeatures2d.SIFT_create(nOctaveLayers=20)
sift2 = cv2.xfeatures2d.SIFT_create(nOctaveLayers=20)


MIN_MATCH_COUNT = 5

# find the keypoints and descriptors with SIFT
kp1, des1 = sift1.detectAndCompute(img1,None)
kp2, des2 = sift2.detectAndCompute(img2,None)
print("model Keypoints found: %s"%(len(kp1)))
print("image Keypoints found: %s"%(len(kp2)))

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
	if m.distance <0.7*n.distance:
		good.append(m)

if len(good)>MIN_MATCH_COUNT:
	src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
	dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

	M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
	matchesMask = mask.ravel().tolist()

	h,w = img1.shape
	pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
	dst = cv2.perspectiveTransform(pts,M)
	img2 = cv2.polylines(img2,[np.int32(dst)],True,255,5, cv2.LINE_AA)

else:
	print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
	matchesMask = None



draw_params = dict(matchColor = (0,255,0), # draw matches in green color
				   singlePointColor = (255,0,0),
				   matchesMask = matchesMask, # draw only inliers
				   flags = 2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

plt.imshow(img3, 'gray'),plt.show()



