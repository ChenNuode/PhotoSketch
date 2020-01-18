import cv2
import json
import numpy as np
import cv2
from json_tricks import dumps

# Load an color image in grayscale

def imgtojson(filepath,display = False, imagewidth=800,imageheight=570):
	
	img = cv2.imread(filepath)

	#display = False
	#imagewidth = 800
	#imageheight = 570

	img = cv2.resize(img, (imagewidth,imageheight) , interpolation = cv2.INTER_AREA)
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	
	countourdata = {}

	colours = [
		([88, 186, 88], [108, 246, 148],"blue"), 
		([114,50,50],[134,255,255],"purple"),
		([0,50,50],[20,255,255],"red"), 
		([65,50,50],[85,255,255],"green"),
		([25,100,100],[35,255,255],"yellow"),
		([0,0,0],[30,70,110],"black"),
	]


	# loop over the colours
	for (lower, upper,mycolour) in colours:
		
		# create NumPy arrays from the boundaries
		lower = np.array(lower, dtype = "uint8")
		upper = np.array(upper, dtype = "uint8")
		smallimage = img.copy()
		
		# find the colors within the specified boundaries and apply
		# the mask
		mask = cv2.inRange(hsv, lower, upper)


		kernel = np.ones((5,5),np.uint8)
		opening = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

		#cv2.imshow("mask",mask)
		cv2.imshow("opening",opening)

		#opening2 = cv2.GaussianBlur(opening,(5, 5),0)
		output = cv2.bitwise_and(img, img, mask = opening)
		
		# show the images
		#cv2.imshow("images", np.hstack([img, output]))
		#cv2.waitKey(0)

		contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

		newcontours = []
		index = 0

		#filter out the anomaly contours
		for c in contours:
			x, y, w, h = cv2.boundingRect(c)
			parent_status = hierarchy[0][index][3]
			
			if ( w * h <= 0.95 * imageheight * imagewidth and w * h > 5) and parent_status == -1:
				newcontours.append(c)
			index += 1

		#show the contours
		if display == True:
			for i in range(len(newcontours)):
				cv2.drawContours(smallimage, newcontours, i, (0,255,0), 1)
				cv2.imshow("hello",smallimage)
				cv2.waitKey(0)
		
		countourdata[mycolour] = newcontours


	blankmap = np.zeros((imageheight,imagewidth), dtype=int)

	for c in countourdata['blue']:
		for pt in c:
			blankmap[pt[0][1]][pt[0][0]] = 1

	for c in countourdata['green']:
		for pt in c:
			blankmap[pt[0][1]][pt[0][0]] = 2

	for c in countourdata['red']:
		for pt in c:
			blankmap[pt[0][1]][pt[0][0]] = 3

	for c in countourdata['purple']:
		for pt in c:
			blankmap[pt[0][1]][pt[0][0]] = 4

	for c in countourdata['yellow']:
		for pt in c:
			blankmap[pt[0][1]][pt[0][0]] = 5

	for c in countourdata['black']:
		for pt in c:
			blankmap[pt[0][1]][pt[0][0]] = 6

	#print(blankmap)
	#np.savetxt("result.txt", blankmap,delimiter=',')

	finaldata = dumps({'mapdata':blankmap}, primitives=True)
	
	with open('1.txt', 'w') as outfile:
		json.dump(finaldata, outfile)


if __name__ == "__main__":
	imgtojson("drawing.jpg")

