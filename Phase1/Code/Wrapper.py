#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code


Author(s): 
Nitin J. Sanket (nitin@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park

Chahat Deep Singh (chahat@terpmail.umd.edu) 
PhD Student in Computer Science,
University of Maryland, College Park
"""

# Code starts here:

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import scipy as sp
from scipy import signal
from scipy import misc
from sklearn.cluster import KMeans
import imutils
import cv2



def gauss1d (sigma, x, diff):
	gaussian = np.exp(-(0.5*(x/sigma)**2))
	if diff == 0:
		pass
	elif diff == 1:
		gaussian = -x/(sigma**2)*gaussian
	elif diff==2:
		gaussian = -1/(sigma**2)*gaussian + gaussian*(x/(sigma**2))**2
	return gaussian

def DG (sigma, theta):
	Sx = np.matrix([[1, 0, -1], 
						[2, 0, -2], 
						[1, 0, -1]])
	Sy = np.matrix([[-1, -2, -1], 
						[0, 0, 0], 
						[1, 2, 1]])
	x, y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
	gx = gauss1d(sigma, x, 0)
	gy = gauss1d(sigma, y, 0)

	gaussian = gx * gy
	dogx = sp.signal.convolve2d(gaussian, Sx, mode='same')
	dogy = sp.signal.convolve2d(gaussian, Sy, mode='same')
	dog = dogx + dogy
	dog = imutils.rotate(dog, theta)
	return dog
	
def LM(sigma_x, theta = 0, diffx = 0, diffy = 0):
	sigma_y = 3*sigma_x
	xr, yr = np.meshgrid(np.linspace(-20,20,100), np.linspace(-20,20,100))
	y = np.cos(theta)*xr + np.sin(theta)*yr
	x = -np.sin(theta)*xr + np.cos(theta)*yr
	gx = gauss1d(sigma_x, x, diffx)
	gy = gauss1d(sigma_y, y, diffy)
	gaussian = gx * gy
	return gaussian

def LoG(sigma):
	x,y = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-7, 7, 100))
	var = np.square(sigma)
	g = (1/np.sqrt(2*np.pi*var))*np.exp( -(x*x + y*y) / (2*var) )
	log = g*((x*x + y*y) - var)/(var**2)
	return log

def Gaussian(sigma, i = 2, j = 2):
	x, y = np.meshgrid(np.linspace(-5,5,50), np.linspace(-5,5,50))
	dst = np.sqrt(x*x + y*y)
	gaussian = np.exp(-32*np.log(2) *((dst)**2 / (2* sigma**2)))
	return gaussian

def resize(img):
	min_val,max_val=img.min(),img.max()
	img = 255.0*(img - min_val)/(max_val - min_val)
	img = img.astype(np.uint8)
	kernel_rz = cv2.resize(img, (100,100))
	return kernel_rz

def Gabor(sigma = 2, theta = 0):
	freq = 0.9/sigma
	xr, yr = np.meshgrid(np.linspace(-0.5, 0.5, 15), np.linspace(-0.5, 0.5, 15))
	x = np.cos(theta)*xr + np.sin(theta)*yr
	y = -np.sin(theta)*xr + np.cos(theta)*yr
	gaussx = gauss1d(sigma, x, 0)
	gaussy = gauss1d(sigma, y, 0)
	gauss = gaussx * gaussy
	gabor = (1/(sigma * np.sqrt(2*np.pi)))*gauss * np.cos(2*np.pi*freq*x)
	return gabor

def half_disk(size):
	mask = np.zeros((size, size))
	radius = size // 2 - size//10
	mask = cv2.circle(mask, (size//2,size//2), radius, (255,255,255), -1)
	for i in range(size):
		for j in range(size):
			if mask[i,j] == 255:
				mask[i,j] = 1
			else:
				mask[i,j] = 0
	for i in range(size//2, size):
		for j in range(size):
			mask[i,j] = 0
	return mask

def texton_map(image, bank, k_size = 64):
	bank_shape = bank.shape
	filtered = []

	for i in range(bank_shape[0]):
		new = sp.signal.convolve2d(image, bank[i], mode='same')
		filtered = [new]+filtered

	filtered = np.array(filtered)
	filtered = np.swapaxes(filtered,0, 2)
	filtered = np.swapaxes(filtered,0, 1)
	shape = filtered.shape
	filtered = filtered.reshape((shape[0]*shape[1]), shape[2])
	texton = np.zeros((shape[0], shape[1]))
	kmeans = KMeans(n_clusters = k_size).fit_predict(filtered)
	kmeans = np.array(kmeans, dtype=np.uint8)
	print(np.max(kmeans))
	texton = kmeans.reshape(shape[0],shape[1])
	return texton

def color_map(image, k_size = 16):
	shape = image.shape
	new_image = image.reshape((shape[0]*shape[1]), shape[2])
	kmeans = KMeans(n_clusters = k_size).fit_predict(new_image)
	kmeans = np.array(kmeans, dtype=np.uint8)
	color_map = kmeans.reshape(shape[0],shape[1])
	return color_map

def brightness_map(image, k_size = 16):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	shape = image.shape
	new_image = image.reshape((shape[0]*shape[1]), 1)
	kmeans = KMeans(n_clusters = k_size).fit_predict(new_image)
	kmeans = np.array(kmeans, dtype=np.uint8)
	brightness_map = kmeans.reshape(shape[0],shape[1])
	return brightness_map

def chi_square(g, h):
	return (0.5 * ((g-h)**2)/(g+h))

def map_g(image, bank, bins):
	image = image + 1
	imshape = image.shape
	shape = bank.shape
	num = np.arange(0, shape[0], 2)
	chi_sqr = []

	for i in num:
		temp = image * 0
		for j in range(1, bins+1):
			for m in range(imshape[0]):
				for n in range(imshape[1]):
					if image[m,n] == j:
						temp[m,n] = 1
					else:
						temp[m,n] = 0
			g = sp.signal.convolve2d(image, bank[i], mode='same')
			h = sp.signal.convolve2d(image, bank[i+1], mode='same')
			chi_sqr = chi_sqr + [chi_square(g,h)]

	chi_sqr = np.array(chi_sqr)
	chi_sqr = np.sum(chi_sqr, 0) / chi_sqr.shape[0]
	return chi_sqr

def scale(x):
	x = (x/np.max(x))*255
	return x


def main():

	path = './BSDS500/'
	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""

	orientations = np.linspace(-45, 315, 16)
	sigma = np.array([1,1.414])
	DoG_bank = []
	fig = plt.figure(figsize=(2, 16), dpi=150)
	d = 1

	for i in sigma:
		for j in orientations:
			dog = DG(i, j)
			DoG_bank = [dog] + DoG_bank
			dog = resize(dog)
			fig.add_subplot(2, 16, d)
			plt.imshow(dog, cmap='gray', vmin=0, vmax=255)
			plt.axis('off')
			d = d + 1
	fig.set_size_inches(5, 1, forward=True)
	plt.show()

	"""
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
	"""

	sigma = np.array([2.828, 2, 1.414])
	g_sigma = np.array([0.7, 1.414, 2.828, 4])
	log_sigma = np.array([0.7, 1, 1.414, 2, 2.4, 2.828, 3.5, 4])
	orientations = np.linspace(0, 5/6*np.pi, 6)
	LM_bank = []
	fig = plt.figure(figsize=(4, 12), dpi=150)
	d = 1

	for i in sigma:
		for j in orientations:
			DoG = LM(i, j, 1)
			LM_bank = [DoG] + LM_bank
			DoG = resize(DoG)
			fig.add_subplot(4, 12, d)
			plt.imshow(DoG, cmap='gray', vmin=0, vmax=255)
			plt.axis('off')
			d = d + 1
		for j in orientations:
			DDoG = LM(i, j, 2)
			LM_bank = [DDoG] + LM_bank
			DDoG = resize(DDoG)
			fig.add_subplot(4, 12, d)
			plt.imshow(DDoG, cmap='gray', vmin=0, vmax=255)
			plt.axis('off')
			d = d + 1

	for i in log_sigma:
		log = LoG(i)
		LM_bank = [log] + LM_bank
		log = resize(log)
		fig.add_subplot(4, 12, d)
		plt.imshow(log, cmap='gray', vmin=0, vmax=255)
		plt.axis('off')
		d = d + 1

	for i in g_sigma:
		g = Gaussian(i)
		LM_bank = [g] + LM_bank
		g = resize(g)
		fig.add_subplot(4, 12, d)
		plt.imshow(g, cmap='gray', vmin=0, vmax=255)
		plt.axis('off')
		d = d + 1
	fig.set_size_inches(5, 2, forward=True)
	plt.show()

	"""
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""

	arr = np.array([0, 1, 2, 3])
	# sigma = np.sqrt(2)**arr 
	sigma=np.array([0.15, 0.22, 0.3, 0.45])
	orientations = np.linspace(np.pi/2, -3*np.pi/8, 8)
	filter_bank = []
	fig = plt.figure(figsize=(4, 8), dpi=150)
	d = 1

	for i in arr:
		for j in orientations:
			gabor = Gabor(sigma[i], j)
			filter_bank = [gabor] + filter_bank
			fig.add_subplot(4, 8, d)
			gabor = resize(gabor)
			plt.imshow(gabor, cmap='gray', vmin=0, vmax=255)
			plt.axis('off')
			d = d+1

	fig.set_size_inches(5, 3, forward=True)
	show = plt.show()
	
	filter_bank = DoG_bank + LM_bank + filter_bank
	filter_bank = np.array(filter_bank)

	"""
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""

	size = np.array([10, 20, 30])
	orientations = np.linspace(90, 180+150, 8)
	d = 1
	hd = []
	fig = plt.figure(figsize=(6,8))
	for i in size:
		for j in orientations:
			hd1 = half_disk(i)
			hd1 = imutils.rotate(hd1, j)
			hd2 = imutils.rotate(hd1, 180)
			hd = hd + [hd1] + [hd2]
	
	hd = np.array(hd)
	d = 1
	for i in range(hd.shape[0]):
		fig.add_subplot(6, 8, d)
		d = d + 1
		plt.imshow(hd[i], cmap='gray')
		plt.axis('off')

	plt.show()
	
	"""
	Generate Texton Map
	Filter image using oriented gaussian filter bank
	"""
	img1 = cv2.imread(path+'Images/1.jpg', 0)

	"""
	Generate texture ID's using K-means clustering
	Display texton map and save image as TextonMap_ImageName.png,
	use command "cv2.imwrite('...)"
	"""
	texton = texton_map(img1, filter_bank, 64)
	fig = plt.figure()
	plt.imshow(texton, cmap='Set1')
	plt.show()

	"""
	Generate Texton Gradient (Tg)
	Perform Chi-square calculation on Texton Map
	Display Tg and save image as Tg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""

	tg = map_g(texton, hd, 64)
	fig = plt.figure()
	plt.imshow(tg)
	plt.show()

	"""
	Generate Brightness Map
	Perform brightness binning 
	"""

	image = cv2.imread(path+'Images/1.jpg')
	bm = brightness_map(image)
	fig = plt.figure()
	plt.imshow(bm, cmap='Set1')
	plt.show()

	"""
	Generate Brightness Gradient (Bg)
	Perform Chi-square calculation on Brightness Map
	Display Bg and save image as Bg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""

	bg = map_g(bm, hd, 16)
	fig = plt.figure()
	plt.imshow(bg)
	plt.show()

	"""
	Generate Color Map
	Perform color binning or clustering
	"""

	image = cv2.imread(path+'Images/1.jpg')
	cm = color_map(image, 16)
	fig = plt.figure()
	plt.imshow(cm, cmap='Set1')
	plt.show()
	
	"""
	Generate Color Gradient (Cg)
	Perform Chi-square calculation on Color Map
	Display Cg and save image as Cg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""

	cg = map_g(cm, hd, 16)
	fig = plt.figure()
	plt.imshow(cg)
	plt.show()

	"""
	Read Sobel Baseline
	use command "cv2.imread(...)"
	"""

	sob = cv2.imread(path+'SobelBaseline/1.png')
	sob = cv2.cvtColor(sob, cv2.COLOR_BGR2GRAY)

	"""
	Read Canny Baseline
	use command "cv2.imread(...)"
	"""

	can = cv2.imread(path+'CannyBaseline/1.png')
	can = cv2.cvtColor(can, cv2.COLOR_BGR2GRAY)

	"""
	Combine responses to get pb-lite output
	Display PbLite and save image as PbLite_ImageName.png
	use command "cv2.imwrite(...)"
	"""

	ult = (tg + bg + cg) / 3
	pb = scale(ult) * (0.5*scale(sob) + 0.5*scale(can))
	fig = plt.figure()
	plt.imshow(pb, cmap='gray')
	plt.show()
	cv2.imshow("pb-lite",pb)
	cv2.waitKey(0)
    
if __name__ == '__main__':
    main()