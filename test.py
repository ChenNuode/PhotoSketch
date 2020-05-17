from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os


def BGR2HEX(color):
	return "#{:02x}{:02x}{:02x}".format(int(color[2]), int(color[1]), int(color[0]))

def get_image(image_path):
	img_og = cv2.imread(image_path)
	img = cv2.cvtColor(img_og, cv2.COLOR_RGB2GRAY)
	img = cv2.medianBlur(img,3)
	th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,5)
	#cv2.imshow("lines", th3)
	final = cv2.bitwise_and(img_og,img_og,mask = th3)
	#cv2.imshow("final",final)
	#cv2.waitKey(0)
	final[th3 == 0] = (255, 255, 255)
	return final

def get_colors(image,number_of_colors,show_chart):
	modified_image = cv2.resize(image, (600, 400),interpolation = cv2.INTER_AREA)
	#modified_image = image
	cv2.imshow("test",modified_image)
	cv2.waitKey(0)

	modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)

	
	clf = KMeans(n_clusters = number_of_colors)
	labels = clf.fit_predict(modified_image)
	
	counts = Counter(labels)
	#print(counts)

	center_colors = clf.cluster_centers_
	# We get ordered colors by iterating through the keys
	ordered_colors = [center_colors[i] for i in counts.keys()]
	print(ordered_colors)

	hex_colors = [BGR2HEX(ordered_colors[i]) for i in counts.keys()]
	rgb_colors = [ordered_colors[i] for i in counts.keys()]

	if (show_chart):
		plt.figure(figsize = (8, 6))
		plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
		plt.show()

	return rgb_colors


get_colors(get_image('examples_images/pic4.jpeg'), 5, True)


