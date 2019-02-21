#import tensorflow as tf
#import matplotlib.pyplot as plt
#import numpy as np
#import sys
#from sklearn import linear_model

def dz_dx(x,y):
    return (2*(x-2)) / ((25-(x-2)**2) - ((y - 3)**2))

def dz_dy(x,y):
    return (2*(y-3)) / ((25-(x-2)**2) - ((y - 3)**2))

x_start = 0
y_start = 0
learning_rate = 0.01
max_iterations = 10000
epsilon = 0.000001
history=[(x_start,y_start)]
for i in range(max_iterations-1):
    dW = dz_dx(x_start,y_start)
    db = dz_dy(x_start,y_start)
    x_start1 = x_start - (learning_rate *dW)
    y_start1 = y_start - (learning_rate *db)
    history.append((x_start1,y_start1))
    print(x_start1,y_start1)
    if abs(x_start1 - x_start) <= epsilon and abs(y_start1 - y_start) <= epsilon:
        print("epsilon hit. i= ",i)
        break
    x_start = x_start1
    y_start = y_start1
print(x_start,y_start)    
