import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn import linear_model
class Problem1Base():
    def __init__(self):
        pass

    def create_dataset(self):
        n_samples = 30
        self.x_point = np.linspace(0,20,n_samples)
        self.y_point = 3.7 * self.x_point + 14 + 4 * np.random.randn(n_samples)
        plt.plot(self.x_point, self.y_point,'o')
        plt.savefig("lesson5.png")

class Problem1SK(Problem1Base):
    def __init__(self):
        pass

    def compute_regression(self):
        linreg = linear_model.LinearRegression()
        self.x_point = np.reshape(self.x_point,(len(self.x_point),1))
        linreg.fit(self.x_point,self.y_point,self.y_point)
        return linreg.coef_,linreg.intercept_

class Problem1TF(Problem1Base):
    def __init__(self):
        self.graph = tf.Graph()
        self.RANDOM_SEED = 42
        tf.set_random_seed(self.RANDOM_SEED)
        np.random.seed(self.RANDOM_SEED) 

    def build(self):
        self.X = tf.placeholder(tf.float32)
        self.Y = tf.placeholder(tf.float32)
        self.LR = tf.placeholder(tf.float32)
        self._slope = tf.Variable(tf.random_uniform([1],-1.0,1.0))
        self._intercept = tf.Variable(tf.zeros([1]))
        self._response = self._slope*self.X + self._intercept
        self._cost = tf.reduce_mean(tf.square(self._response - self.Y))
        self._optimizer = tf.train.GradientDescentOptimizer(self.LR).minimize(self._cost)

    @property
    def cost(self):
        return self._cost
    @property
    def optimizer(self):
        return self._optimizer
    @property
    def slope(self):
        return self._slope
    @property
    def intercept(self):
        return self._intercept

    def compute_regression(self):
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            for epoch in range(50000):
    #           for (_x,_y) in zip(self.x_point,self.y_point):
    #               session.run(self._optimizer,feed_dict = {self.LR: 0.0001,self.X : _x, self.Y : _y})
                session.run(self._optimizer,feed_dict = {self.LR: 0.0001,self.X : self.x_point, self.Y : self.y_point})
                if (epoch % 500) == 0:    
                    c = session.run(self._cost, feed_dict = {self.LR: 0.0001,self.X : self.x_point, self.Y : self.y_point})
                    print("Epoch", epoch , ": cost =", c, "slope =", session.run(self._slope), "intercept =", session.run(self._intercept))
            print("Slope = ",session.run(self._slope))
            print("Intercept = ",session.run(self._intercept))

def main():
    problem1 = Problem1SK()
    problem1.create_dataset()
    print(problem1.compute_regression())

    problem1 = Problem1TF()
    problem1.create_dataset()
    problem1.build()
    print(problem1.compute_regression())

if __name__ == "__main__":
    main()
