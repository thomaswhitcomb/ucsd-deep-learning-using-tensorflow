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
        x = np.linspace(0,20,n_samples)
        self.train_x = np.reshape(x,(len(x),1))
        self.train_y = 3.7 * self.train_x + 14 + 4 * np.random.randn(n_samples)
        plt.plot(self.train_x, self.train_y,'o')
        plt.savefig("lesson5.png")
        print(self.train_x)
        print(self.train_y)

    def compute_regression(self):
        linreg = linear_model.LinearRegression()
        #print("train_x",self.train_x.shape)
        #print("train_y",self.train_y.shape)
        linreg.fit(self.train_x,self.train_y)
        print(linreg.intercept_)
        print(linreg.coef_)




def main():
    problem1 = Problem1()
    problem1.create_dataset()
    problem1.compute_regression()

if __name__ == "__main__":
    main()
