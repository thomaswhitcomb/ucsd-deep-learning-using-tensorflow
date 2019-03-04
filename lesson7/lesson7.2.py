import numpy as np
from scipy import signal as sg
import tensorflow as tf

image = np.genfromtxt('Problem#2-Image.csv', delimiter=',')
filtr = np.genfromtxt('Problem#2-Filter Gaussian.csv',delimiter=",")
 
sg_full = sg.convolve(image,filtr,"full")
sg_valid = sg.convolve(image,filtr,"valid")

print("SCIPY FULL")
print(sg_full)
print("SCIPY VALID")
print(sg_valid)

image_list = tf.constant(image)
image = tf.reshape(image_list,[1,image_list.shape[0],image_list.shape[1],1])

filtr_list = tf.constant(filtr)
filtr = tf.reshape(filtr_list,[filtr_list.shape[0],filtr_list.shape[1],1,1])

tf_valid = tf.nn.conv2d(image,filtr,strides=[1,1,1,1], padding='VALID')
tf_same = tf.nn.conv2d(image,filtr,strides=[1,1,1,1], padding='SAME')

with tf.Session() as sess:
    sess.run(image)
    sess.run(filtr)
    print("TF SAME")
    print(sess.run(tf_same))
    print("TF VALID")
    print(sess.run(tf_valid))
