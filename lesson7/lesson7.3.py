import numpy as np
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt 
from PIL import Image
from scipy import ndimage 
import sys
import math

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

def make_filter(n):

  filtr = [[1 for i in range(n)] for j in range(n)]
  filtr[math.ceil(n/2)][math.ceil(n/2)] = 1-(n ** 2)
  return filtr

def filter_image(image,filtr_size,name):
    filtr = make_filter(filtr_size)
    grad = signal.convolve2d(image,filtr, mode='same', boundary='symm')
    misc.imsave(name,np.absolute(grad))

filter_image(image_array,7,"auto_5x5.png")
filter_image(image_array,11,"auto_11x11.png")
filter_image(image_array,15,"auto_15x15.png")
filter_image(image_array,29,"auto_29x29.png")
filter_image(image_array,39,"auto_39x39.png")
