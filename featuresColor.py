#!/usr/bin/env python

import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

def getRGBS(img, PLOT = False):

	image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

	# grab the image channels, initialize the tuple of colors,
	# the figure and the flattened feature vector	
	features = []
	featuresSobel = []
	Grayscale = cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY)
	histG = cv2.calcHist([Grayscale], [0], None, [8], [0, 256])
	histG = histG / histG.sum()
	features.extend(histG[:,0].tolist())


	grad_x = np.abs(cv2.Sobel(Grayscale, cv2.CV_16S, 1, 0, ksize = 3, scale = 1, delta = 0, borderType = cv2.BORDER_DEFAULT))
	grad_y = np.abs(cv2.Sobel(Grayscale, cv2.CV_16S, 0, 1, ksize = 3, scale = 1, delta = 0, borderType = cv2.BORDER_DEFAULT))
	abs_grad_x = cv2.convertScaleAbs(grad_x)
	abs_grad_y = cv2.convertScaleAbs(grad_y)
	dst = cv2.addWeighted(abs_grad_x,0.5,abs_grad_y,0.5,0)
	histSobel = cv2.calcHist([dst], [0], None, [8], [0, 256])
	histSobel = histSobel / histSobel.sum()
	features.extend(histSobel[:,0].tolist())

	Fnames = []
	Fnames.extend(["Color-Gray"+str(i) for i in range(8)])
	Fnames.extend(["Color-GraySobel"+str(i) for i in range(8)])

	return features, Fnames
