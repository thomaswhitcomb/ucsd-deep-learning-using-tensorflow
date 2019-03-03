import numpy as np
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt 
from PIL import Image

im = Image.open('01 Lady.png')
fig, aux = plt.subplots(figsize=(512,512)) 
fig, aux = plt.subplots() 
aux.imshow(im, cmap='gray')
misc.imsave("filtered_lady_start.png",im)

image_gr = im.convert("L")
arr = np.asarray(image_gr)

filtr = np.genfromtxt('filter2.csv',delimiter=",")
grad = signal.convolve2d(arr,filtr, mode='same', boundary='symm')
misc.imsave("filtered_lady_2.png",np.absolute(grad))

filtr = np.genfromtxt('filter1.csv',delimiter=",")
grad = signal.convolve2d(arr,filtr, mode='same', boundary='symm')
misc.imsave("filtered_lady_1.png",np.absolute(grad))

filtr = np.genfromtxt('filter2.csv',delimiter=",")
grad = signal.convolve2d(grad,filtr, mode='same', boundary='symm')
misc.imsave("filtered_lady_3.png",np.absolute(grad))

#fig, aux = plt.subplots(figsize=(512,512)) 
#fig, aux = plt.subplots() 
#aux.imshow(np.absolute(grad), cmap='gray')
