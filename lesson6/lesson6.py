import tensorflow as tf
import matplotlib.pyplot as plt
import sys
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

def dz_dx(x,y):
    return (2*(x-2)) / ((25-(x-2)**2) - ((y - 3)**2))

def dz_dy(x,y):
    return (2*(y-3)) / ((25-(x-2)**2) - ((y - 3)**2))

def problem1():
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
            return x_start1, y_start1    
        x_start = x_start1
        y_start = y_start1
    return None,None

def problem2():

    RANDOM_SEED = 42
    tf.set_random_seed(RANDOM_SEED)
    n_samples = 30
    train_x = np.linspace(0,20,n_samples)
    train_y = 3.7 * train_x + 14 + 4 * np.random.randn(n_samples)
    plt.plot(train_x, train_y,'o')
    linreg = linear_model.LinearRegression()
    train_x = np.reshape(train_x,(len(train_x),1))
    linreg.fit(train_x,train_y,train_y)
    return linreg.coef_,linreg.intercept_

def main():
    print("Problem 1")
    print(problem1())
    print(problem2())


if __name__ == "__main__":
    main()
