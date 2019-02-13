import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
class Graph1():
    def __init__(self):
        self.graph = tf.Graph()
    def build(self):
        with self.graphr.as_Default():
            _slope = tf.Variable(tf.random_uniform([1],-1.0,1.0))
            _intercept = tf.Variable(tf.zeros([1]))
            _response = _slope*x_point + intercept
            self._cost = tf.reduce_mean(tf.square(_response - y_point))
            self._optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cost)
    @property
    def cost(self):
        return self._cost
    @property
    def optimizer(self):
        return self._optimizer


class Problem1SK():
    def __init__(self):
        pass

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

class Problem1TF():
    def __init__(self):
        self.RANDOM_SEED = 42
        tf.set_random_seed(self.RANDOM_SEED)

    def create_dataset(self):
        number_of_points = 500
        self.x_point = []
        self.y_point = []
        m = 0.22
        c = 0.78
        for i in range(number_of_points):
            x = np.random.normal (0.0, 0.5)
            y = m*x + c + np.random.normal(0.0,0.1)
            self.x_point.append([x])
            self.y_point.append([y])

    def compute_regression(self):
        init = tf.global_variables_initializer()
        session.run(init)
        for epoch in range(30):
            session.run(optimizer)
            if (epoch % 5) == 0:

def main():
    problem1 = Problem1()
    problem1.create_dataset()
    problem1.compute_regression_with_sklearn()

if __name__ == "__main__":
    main()
