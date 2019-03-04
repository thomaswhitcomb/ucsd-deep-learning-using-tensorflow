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

filtr1 = np.genfromtxt('filter1.csv',delimiter=",")
grad = signal.convolve2d(image_array,filtr1, mode='same', boundary='symm')
misc.imsave("filtered_lady_1.png",np.absolute(grad))

filtr2 = np.genfromtxt('filter2.csv',delimiter=",")
grad = signal.convolve2d(image_array,filtr2, mode='same', boundary='symm')
misc.imsave("filtered_lady_2.png",np.absolute(grad))

filtr1plus2 = np.genfromtxt('filter2.csv',delimiter=",")
grad = signal.convolve2d(image_array,filtr1+filtr2, mode='same', boundary='symm')
misc.imsave("filtered_lady_1plus2.png",np.absolute(grad))

filtr_5x5 = np.array([
    [-1, -1, -1, -1, -1],
    [-1,  1,  2,  1, -1],
    [-1,  2,  4,  2, -1],
    [-1,  1,  2,  1, -1],
    [-1, -1, -1, -1, -1]
])

filtr_7x7 = np.array([
    [ 0, 0,-1,-1,-1, 0, 0 ],
    [ 0,-1,-3,-3,-3,-1, 0 ],
    [-1,-3, 0, 7, 0,-3,-1],
    [-1,-3, 7,24, 7,-3,-1],
    [-1,-3, 0, 7, 0,-3,-1],
    [ 0,-1,-3,-3,-3,-1, 0],
    [ 0, 0,-1,-1,-1, 0, 0 ]])

filtr_9x9 = np.array([
    [ 0, 1, 1, 2, 2, 2, 1, 1, 0],
    [ 1, 2, 4, 5, 5, 5, 4, 2, 1],
    [ 1, 4, 5, 3, 0, 3, 5, 4, 1],
    [ 2, 5, 3, -12, -24, -12, 3, 2, 1],
    [ 2, 5, 0, -24, -40, -24, 0, 5, 2],
    [ 2, 5, 3, -12, -24, -12, 3, 2, 1],
    [ 1, 4, 5, 3, 0, 3, 5, 4, 1],
    [ 1, 2, 4, 5, 5, 5, 4, 2, 1],
    [ 0, 1, 1, 2, 2, 2, 1, 1, 0]])

filtr_13x13 = np.array([
    [ 0, 0, 0, 0, 0,-1, -1,-1,  0, 0,  0, 0, 0],
    [ 0, 0, 0,-1,-1,-2, -2,-2, -1,-1,  0, 0, 0],
    [ 0, 0,-2,-2,-3,-3, -4,-3, -3,-2, -2, 0, 0],
    [ 0,-1,-2,-3,-3,-3, -2,-3, -3,-3, -2,-1, 0],
    [ 0,-1,-3,-3,-1, 4,  6, 4, -1,-3, -3,-1, 0],
    [-1,-2,-3,-3, 4,14, 19,14,  4,-3, -3,-2,-1],
    [-1,-2,-4,-2, 6,19, 24,19,  6,-2, -4,-2,-1],
    [-1,-2,-3,-3, 4,14, 19,14,  4,-3, -3,-2,-1],
    [ 0,-1,-3,-3,-1, 4,  6, 4, -1,-3, -3,-1, 0],
    [ 0,-1,-2,-3,-3,-3, -2,-3, -3,-3, -2,-1, 0],
    [ 0, 0,-2,-2,-3,-3, -4,-3, -3,-2, -2, 0, 0],
    [ 0, 0, 0,-1,-1,-2, -2,-2, -1,-1,  0, 0, 0],
    [ 0, 0, 0, 0, 0,-1, -1,-1,  0, 0,  0, 0, 0]])
            
grad = signal.convolve2d(image_array,filtr_5x5, mode='same', boundary='symm')
misc.imsave("filtered_lady_5x5.png",np.absolute(grad))

grad = signal.convolve2d(image_array,filtr_7x7, mode='same', boundary='symm')
misc.imsave("filtered_lady_7x7.png",np.absolute(grad))

grad = signal.convolve2d(image_array,filtr_9x9, mode='same', boundary='symm')
misc.imsave("filtered_lady_9x9.png",np.absolute(grad))

grad = signal.convolve2d(image_array,filtr_13x13, mode='same', boundary='symm')
misc.imsave("filtered_lady_13x13.png",np.absolute(grad))

filtr_15x15 = [[1 for i in range(15)] for j in range(15)]
filtr_15x15[8][8] = -224
grad = signal.convolve2d(image_array,filtr_15x15, mode='same', boundary='symm')
misc.imsave("filtered_lady_15x15.png",np.absolute(grad))
