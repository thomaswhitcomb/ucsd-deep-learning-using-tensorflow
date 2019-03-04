import numpy as np
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt 
from PIL import Image
import scipy.ndimage.filters

im = Image.open('01 Lady.png')
misc.imsave("filtered_lady_start.png",im)

image_gr = im.convert("L")
image_array = np.asarray(image_gr)

filtr = np.genfromtxt('filter2.csv',delimiter=",")
grad = signal.convolve2d(image_array,filtr, mode='same', boundary='symm')
misc.imsave("filtered_lady_2.png",np.absolute(grad))

filtr = np.genfromtxt('filter1.csv',delimiter=",")
grad = signal.convolve2d(image_array,filtr, mode='same', boundary='symm')
misc.imsave("filtered_lady_1.png",np.absolute(grad))

filtr = np.genfromtxt('filter2.csv',delimiter=",")
grad = signal.convolve2d(grad,filtr, mode='same', boundary='symm')
misc.imsave("filtered_lady_3.png",np.absolute(grad))

filtr_5x5 = np.array([
    [-1, -1, -1, -1, -1],
    [-1,  1,  2,  1, -1],
    [-1,  2,  4,  2, -1],
    [-1,  1,  2,  1, -1],
    [-1, -1, -1, -1, -1]
])
grad = signal.convolve2d(image_array,filtr_5x5, mode='same', boundary='symm')
misc.imsave("filtered_lady_5x5.png",np.absolute(grad))

filtr_15x15 = np.array([
[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
[-1,-1,-1,-1,-1,-1,-1,24,-1,-1,-1,-1,-1,-1,-1],
[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]])

grad = signal.convolve2d(image_array,filtr_15x15, mode='same', boundary='symm')
misc.imsave("filtered_lady_15x15.png",np.absolute(grad))

