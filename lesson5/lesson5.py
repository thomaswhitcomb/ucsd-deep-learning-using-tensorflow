import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn import linear_model
class Problem1Base():
    def __init__(self):
        pass

    def create_dataset(self):
        n_samples = 30
        self.train_x = np.linspace(0,20,n_samples)
        self.train_y = 3.7 * self.train_x + 14 + 4 * np.random.randn(n_samples)
        plt.plot(self.train_x, self.train_y,'o')
        plt.savefig("lesson5.png")



class Problem1SK(Problem1Base):
    def __init__(self):
        pass

    def compute_regression(self):
        linreg = linear_model.LinearRegression()
        self.train_x = np.reshape(self.train_x,(len(self.train_x),1))
        #print("train_x",self.train_x.shape,self.train_x)
        #print("train_y",self.train_y.shape,self.train_y)
        linreg.fit(self.train_x,self.train_y,self.train_y)
        return linreg.coef_,linreg.intercept_

class Problem1TF(Problem1Base):
    def __init__(self):
        self.graph = tf.Graph()
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

    def build(self):
        with self.graph.as_default():
            self._slope = tf.Variable(tf.random_uniform([1],-1.0,1.0))
            self._intercept = tf.Variable(tf.zeros([1]))
            _response = self._slope*self.x_point + self._intercept
            self._cost = tf.reduce_mean(tf.square(_response - self.y_point))
            self._optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(self._cost)

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
        with tf.Session(graph=self.graph) as session:
            init = tf.global_variables_initializer()
            session.run(init)
            for epoch in range(30):
                session.run(self.optimizer)
                if (epoch % 5) == 0:
                    pass
            print("Slope = ",session.run(self.slope))
            print("Intercept = ",session.run(self.intercept))

def main():
    problem1 = Problem1SK()
    problem1.create_dataset()
    print(problem1.compute_regression())
    sys.exit()
    problem1 = Problem1TF()
    problem1.create_dataset()
    problem1.build()
    problem1.compute_regression()

if __name__ == "__main__":
    main()
