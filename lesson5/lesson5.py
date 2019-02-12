import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

class Problem1():
    def __init__(self):
        self.RANDOM_SEED = 42
        tf.set_random_seed(self.RANDOM_SEED)

    def create_dataset(self):
        n_samples = 30
        self.train_x = np.linspace(0,20,n_samples)
        self.train_y = 3.7 * self.train_x + 14 + 4 * np.random.randn(n_samples)
        plt.plot(self.train_x, self.train_y,'o')
        plt.savefig("lesson5.png")

    def compute_regression(self):
        linreg = linear_model.LinearRegression()
        self.train_x = np.reshape(self.train_x,(len(self.train_x),1))
        print("train_x",self.train_x.shape,self.train_x)
        print("train_y",self.train_y.shape,self.train_y)
        linreg.fit(self.train_x,self.train_y,self.train_y)
        print("intercept",linreg.intercept_)
        print("slope",linreg.coef_)




def main():
    problem1 = Problem1()
    problem1.create_dataset()
    problem1.compute_regression()

if __name__ == "__main__":
    main()
