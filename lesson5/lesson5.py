import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn import linear_model

def scale(t):
    tMin = t.min(axis=0)
    tMax = t.max(axis=0)
    return (t-tMin)/(tMax-tMin)

class Problem1Base():
    def __init__(self):
        tf.reset_default_graph()

    def create_dataset(self):
        n_samples = 30
        self.x_point = np.linspace(0,20,n_samples)

        self.y_point = 3.7 * self.x_point + 14 + 4 * np.random.randn(n_samples)

        plt.plot(self.x_point, self.y_point,'o')
        plt.savefig("lesson5.png")

class Problem1SK(Problem1Base):
    def __init__(self):
        self.RANDOM_SEED = 42
        np.random.seed(self.RANDOM_SEED) 

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
        self.slope = tf.Variable(tf.random_uniform([1],-1.0,1.0))
        self.intercept = tf.Variable(tf.zeros([1]))
        self.response = self.slope*self.X + self.intercept
        self.cost = tf.reduce_mean(tf.square(self.response - self.Y))
        self.optimizer = tf.train.GradientDescentOptimizer(self.LR).minimize(self.cost)

    def compute_regression(self,epochs,learning_rate):
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            for epoch in range(epochs):
                session.run(
                        self.optimizer,feed_dict = {
                            self.LR: learning_rate,
                            self.X : self.x_point, 
                            self.Y : self.y_point})
                if (epoch % (epochs/20)) == 0:
                    c = session.run(
                            self.cost, feed_dict = {
                                self.LR: learning_rate,
                                self.X : self.x_point, 
                                self.Y : self.y_point})
                    print("Epoch:", epoch , "Cost:", c, "Slope:", 
                            session.run(self.slope), 
                            "Intercept:", 
                            session.run(self.intercept))
            c = session.run(self.cost, feed_dict = {
                self.LR: learning_rate,
                self.X : self.x_point, 
                self.Y : self.y_point})
            print("Epoch:", epoch , "Cost:", c, "Slope:", 
                    session.run(self.slope), 
                    "Intercept:", 
                    session.run(self.intercept))
            return c,session.run(self.slope),session.run(self.intercept)

class Problem2Base():
    def __init__(self):
        tf.reset_default_graph()

    def create_dataset(self):
        f = open("00 kc_house_data.csv")
        f.readline()
        dataset = np.genfromtxt(fname = f, delimiter = ',',usecols=(2,3,5))
        self.predictors = dataset[:,1:3] 
        predictorsMin = self.predictors.min(axis=0)
        predictorsMax = self.predictors.max(axis=0)
        self.predictors_scaled = scale(self.predictors)
        self.response = dataset[:,0:1]
        responseMin = self.response.min(axis=0)
        responseMax = self.response.max(axis=0)
        self.response_scaled = scale(self.response)

class Problem2SK(Problem2Base):
    def __init__(self):
        self.RANDOM_SEED = 42
        np.random.seed(self.RANDOM_SEED) 

    def compute_regression(self):
        linreg = linear_model.LinearRegression()
        linreg.fit(self.predictors_scaled,self.response_scaled)
        return linreg.coef_,linreg.intercept_

class Problem2TF(Problem2Base):
    def __init__(self):
        self.RANDOM_SEED = 42
        tf.set_random_seed(self.RANDOM_SEED)
        np.random.seed(self.RANDOM_SEED) 

    def build(self):
        self.X1 = tf.placeholder(tf.float32)
        self.X2 = tf.placeholder(tf.float32)
        self.Y = tf.placeholder(tf.float32)

        self.LR = tf.placeholder(tf.float32)

        self.slope1 = tf.Variable([0],dtype=tf.float32,name="weight1")
        self.slope2 = tf.Variable([0],dtype=tf.float32,name="weight2")
        self.intercept = tf.Variable(tf.zeros([1]),dtype=tf.float32,name="bias")
        self.response = self.slope1*self.X1 + self.slope2*self.X2 + self.intercept
        self.cost = tf.reduce_mean(tf.square(self.response - self.Y))
        self.optimizer = tf.train.GradientDescentOptimizer(self.LR).minimize(self.cost)

    def compute_regression(self,epochs,learning_rate):
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            for epoch in range(epochs):
                session.run(
                        self.optimizer,feed_dict = {
                            self.LR: learning_rate,
                            self.X1:self.predictors_scaled[:,0:1],
                            self.X2:self.predictors_scaled[:,1:2],
                            self.Y:self.response_scaled})
                if not epoch % 100000:
                    c = session.run(
                            self.cost, feed_dict = {
                                self.LR:learning_rate,
                                self.X1:self.predictors_scaled[:,0:1],
                                self.X2:self.predictors_scaled[:,1:2],
                                self.Y:self.response_scaled})
                    print("Epoch:", epoch , "Cost:", c, "Slope1:", 
                            session.run(self.slope1),
                            "Slope2:",
                            session.run(self.slope2), 
                            "Intercept:", 
                            session.run(self.intercept))
            c = session.run(
                    self.cost, feed_dict = {
                        self.LR: learning_rate,
                        self.X1 : self.predictors_scaled[:,0:1],
                        self.X2 : self.predictors_scaled[:,1:2], 
                        self.Y : self.response_scaled} )
            print("Epoch:", epoch , "Cost:", c, "Slope1:", 
                    session.run(self.slope1),
                    "Slope2:",
                    session.run(self.slope2), 
                    "Intercept:", 
                    session.run(self.intercept))
            
            return c,session.run(self.slope1),session.run(self.slope2),session.run(self.intercept)


def main():

    print("Problem 1 - SKLearn")
    print("====================")
    problem1 = Problem1SK()
    problem1.create_dataset()
    slope,intercept = problem1.compute_regression()
    print(slope,intercept)
    assert ("%.8f" % slope[0]) == "3.54942311"
    assert ("%.15f" % intercept) == "14.841075232789862"

    print("\nProblem 1, solution 1 - Tensorflow")
    print("====================================")
    print("   Epoch 50000, Learning rate 0.0001")
    problem1 = Problem1TF()
    problem1.create_dataset()
    problem1.build()
    cost,slope,intercept = problem1.compute_regression(50000,0.0001)
    print(cost,slope,intercept)
    assert ("%.5f" % cost) == "11.36642"
    assert ("%.7f" % slope[0]) == "3.5775077"
    assert ("%.6f" % intercept[0]) == "14.184714"

    print("\nProblem 1, solution 2 - Tensorflow")
    print("====================================")
    print("   Epoch 5000, Learning rate 0.001")
    problem1 = Problem1TF()
    problem1.create_dataset()
    problem1.build()
    cost,slope,intercept = problem1.compute_regression(5000,0.001)
    print(cost,slope,intercept)
    assert ("%.6f" % cost) == "11.365601"
    assert ("%.7f" % slope[0]) == "3.5774026"
    assert ("%.6f" % intercept[0]) == "14.186133"

    print("Problem 2 - SKLearn")
    print("====================")
    problem2 = Problem2SK()
    problem2.create_dataset()
    slope,intercept = problem2.compute_regression()
    print(slope,intercept)

    print("Problem 2 - Tensorflow")
    print("======================")
    print("   Epoch 1000000, Learning rate 0.01")
    problem2 = Problem2TF()
    problem2.create_dataset()
    problem2.build()
    cost,slope1,slope2,intercept = problem2.compute_regression(1000000,0.01)

if __name__ == "__main__":
    main()
