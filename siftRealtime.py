import numpy as np
import cv2
from matplotlib import pyplot as plt

def detectSift(outputImg, grayKp, grayDes, modelKp, modelDes,modelShape, sift):

	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 50)

	try:
		flann = cv2.FlannBasedMatcher(index_params, search_params)
		matches = flann.knnMatch(modelDes,grayDes,k=2)
	except:
		return []

	# store all the good matches as per Lowe's ratio test.
	good = []
	for m,n in matches:
		if m.distance <0.7*n.distance:
			good.append(m)

	MIN_MATCH_COUNT = 10
	if len(good)>MIN_MATCH_COUNT:
		src_pts = np.float32([ modelKp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ grayKp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
		matchesMask = mask.ravel().tolist()

		h,w = modelShape
		h = h
		w = w
		pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
		try:
			dst = cv2.perspectiveTransform(pts,M)
			# outputImg = cv2.polylines(outputImg,[np.int32(dst)],True,(255,255,255),5, cv2.LINE_AA)
			return [np.int32(dst)]
		except:
			pass
	return []
def getImageFeatures(img, sift):
	kp2, des2 = sift.detectAndCompute(grayImg,None)
	return kp2, des2

def drawBoundingBox(outputImg,points, color=(0,255,0), thickness=2):
	if(len(points) > 0):
		outputImg = cv2.polylines(outputImg,points,True,color,thickness, cv2.LINE_AA)
	return outputImg

def getHeadModel(sift):
	img1 = cv2.imread('headModel/front.png',0)
	img1 = cv2.GaussianBlur(img1,(41,41),0)
	kp1, des1 = sift.detectAndCompute(img1,None)
	return kp1,des1,img1.shape

def getBodyFrontModel(sift):
	img1 = cv2.imread('bodyModel/front.png',0)
	img1 = cv2.GaussianBlur(img1,(41,41),0)
	kp1, des1 = sift.detectAndCompute(img1,None)
	return kp1,des1,img1.shape
def getBodyBackModel(sift):
	img1 = cv2.imread('bodyModel/back.png',0)
	img1 = cv2.GaussianBlur(img1,(31,31),0)
	kp1, des1 = sift.detectAndCompute(img1,None)
	return kp1,des1,img1.shape
def getArmModel():
	img1 = cv2.imread('armModel/front.png',1)
	hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)

	# maskBlue = cv2.inRange(hsv, (50,0,0),(255,255,255))
	# maskBlue = cv2.bitwise_not(maskBlue)

	# img1 = cv2.bitwise_and(img1,img1,mask=maskBlue)
	img1[np.where((img1==[0,0,0]).all(axis=2))] = [255,255,255];
	edges = cv2.Canny(img1,100,1000)
	kernel = np.ones((3,3))
	edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel,iterations=10)
	cv2.imshow("edges2", edges)
	im2,contours,hierarchy = cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	# print(len(contours))
	# print(contours)
	 #find the biggest area
	c = max(contours, key = cv2.contourArea)
	im2 = cv2.drawContours(img1, [c], 0, (255,0,255), 3)

	# cv2.imshow("arms", im2)
	return [c]

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
def removeBackground(img):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	# # define range of blue color in HSV
	# # # Threshold the HSV image to get only red
	mask1 = cv2.inRange(hsv, (0,100,75), (5,255,255))
	mask2 = cv2.inRange(hsv, (175,93,75), (180,210,255))
	# maskBlue = cv2.inRange(hsv, (50,0,0),(255,255,255))

	mask = cv2.bitwise_or(mask1,mask2)
	# mask = cv2.bitwise_or(mask,maskBlue)
	kernel = np.ones((5,5))
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,iterations = 5)
	kernel = np.ones((10,10))
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,iterations = 1)
	output = cv2.bitwise_and(img,img,mask=mask)

	output[np.where((output==[0,0,0]).all(axis=2))] = [255,255,255];
	cv2.imshow("background removed", output)

	return output

def detectArms(outputImg, testImg, armContours):
	edges = cv2.Canny(testImg,400,800)
	# kernel = np.ones((10,10))
	# edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel,iterations=3)
	cv2.imshow("edges1",edges)
	im2,contours, hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


	##Look for only yellow component in arms
	hsv = cv2.cvtColor(testImg, cv2.COLOR_BGR2HSV)
	yellowMask = cv2.inRange(hsv, (5,0,0), (28,255,255))
	kernel = np.ones((3,3))
	yellowMask = cv2.erode(yellowMask, kernel,iterations=2)
	yellowIm, yellowContours,hierarchy = cv2.findContours(yellowMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	yellowCentroids = []
	for c in yellowContours:
		cx = 0
		cy = 0
		for p in c:
		    cx += p[0][0]
		    cy += p[0][1]
		cx = int(cx/len(c))
		cy = int(cy/len(c))		
		yellowCentroids.append((cx,cy))

	# yellowMask = cv2.drawContours(yellowMask, yellowContours, -1, (0,0,255), 3)
	cv2.imshow("yelow mask", yellowMask)
	# yellowOnly = cv2.bitwise_and(testImg,testImg,mask=yellowMask)
	

	# print(len(yellowCentroids))
 

 	armCandidates=[]
 	legCandidates=[]
 	for c in contours:
 		addedArm = False
 		for centroid in yellowCentroids:
 			if cv2.pointPolygonTest(c, centroid, False) >= 0 and addedArm == False: #the contours contains a yellow centroid
 				armCandidates.append(c)
 				addedArm = True
 				break
		if(not addedArm):
			legCandidates.append(c)

	print("found %s arm candidates"%(len(armCandidates)))
	print("found %s leg candidates"%(len(legCandidates)))

	arms = findBestShapeMatches(armCandidates, armContours[0],2)
	legs = findBestShapeMatches(legCandidates, armContours[0],2)

	cv2.imshow("seg",cv2.drawContours(testImg, contours,-1,(0,0,255),3))
	# outputImg = cv2.drawContours(outputImg, [c], 0, (255,0,255), 3)

	ouputImg = cv2.drawContours(outputImg, arms, -1, (0,0,255), 3)
	ouputImg = cv2.drawContours(outputImg, legs, -1, (255,0,255), 3)


	# # define range of blue color in HSV
	# low_tresh = (109,41,33)
	# up_thresh = (147,133,255)
	# # # Threshold the HSV image to get only red
	# mask = cv2.inRange(testImg, low_tresh, up_thresh)
	#cv2.imshow("edges", edges)
	return outputImg
	# cv2.imshow("mask",mask)
	# cv2.waitKey(0)

def findBestShapeMatches(candidates, modelContour,numTotal):
	matches = []
	matchScores = []

	
	for i in range(len(candidates)):
		cnt = candidates[i]
		matchScores.append(cv2.matchShapes(cnt, modelContour,1,0.0))

	print(matchScores)
	for i in range(min(numTotal, len(candidates))):
		bestIndex = np.argmin(matchScores)
		matches.append(candidates[i])
		matchScores[i] = 9999

	return matches




sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers=3, edgeThreshold=10)
sift2 = cv2.xfeatures2d.SIFT_create(nOctaveLayers=10, edgeThreshold=10)
headKp, headDes, headShape = getHeadModel(sift)
bodyKp, bodyDes, bodyShape = getBodyFrontModel(sift)
backKp, backDes, backShape = getBodyBackModel(sift2)
armContours = getArmModel()
print(headShape)
cam= cv2.VideoCapture(1)

try:
	while True:
		ret, testImg = cam.read()
		testImg = cv2.resize(testImg,(0,0),fx=.5, fy=.5)
		robotOnly = removeBackground(testImg)
		grayImg = 0
		grayImg = cv2.cvtColor(testImg, grayImg, cv2.COLOR_BGR2GRAY);
		outputImg = np.copy(testImg)
		grayKp, grayDes = getImageFeatures(grayImg, sift)
		headPoints = detectSift(outputImg, grayKp, grayDes, headKp, headDes, headShape, sift)
		bodyPoints = detectSift(outputImg, grayKp, grayDes, bodyKp, bodyDes, bodyShape, sift)

		if(len(bodyPoints) > 0):
			robotSegmented = cv2.fillPoly(robotOnly, bodyPoints,color=(255,255,255))
			robotSegmented = cv2.fillPoly(robotOnly, headPoints,color=(255,255,255))

			outputImg = detectArms(outputImg, robotSegmented, armContours)
		# outputImg = detectSift(outputImg, grayKp, grayDes, backKp, backDes, backShape, sift2)
		#outputImg = detectArms(outputImg,testImg,armContours)

		outputImg = drawBoundingBox(outputImg, headPoints, color=(255,255,0))
		outputImg = drawBoundingBox(outputImg, bodyPoints)


		cv2.imshow("output",outputImg)
		cv2.waitKey(3)

except KeyboardInterrupt:
	print('interrupted!')
