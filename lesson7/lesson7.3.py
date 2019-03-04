import numpy as np
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt 
from PIL import Image
from scipy import ndimage 
import sys

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

filtr2 = np.genfromtxt('filter2.csv',delimiter=",")
filtr1 = np.genfromtxt('filter1.csv',delimiter=",")
filtr3 = [[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]
filtr4 = [[0,-1,0],[-1,5,-1],[0,-1,0]]

x1 = np.hstack((filtr1,filtr2,filtr3,filtr2,filtr1))
x2 = np.hstack((filtr1,filtr2,filtr3,filtr2,filtr1))
x3 = np.hstack((filtr1,filtr2,filtr3,filtr2,filtr1))
x4 = np.hstack((filtr1,filtr2,filtr3,filtr2,filtr1))
x5 = np.hstack((filtr1,filtr2,filtr3,filtr2,filtr1))
X = np.vstack((x1,x2,x3,x4,x5))

I = np.array([
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,24,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
L1 = ndimage.laplace(I)
grad = signal.convolve2d(image_array,L1, mode='same', boundary='symm')
misc.imsave("lady_L1.png",np.absolute(grad))

I = np.array([
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,32,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

L2 = ndimage.laplace(I)
grad = signal.convolve2d(image_array,L2, mode='same', boundary='symm')
misc.imsave("lady_L2.png",np.absolute(grad))
