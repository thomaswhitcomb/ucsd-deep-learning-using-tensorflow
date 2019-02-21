import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
import sys

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
        #print(x_start1,y_start1)
        if abs(x_start1 - x_start) <= epsilon and abs(y_start1 - y_start) <= epsilon:
            print("epsilon hit. i= ",i)
            return x_start1, y_start1    
        x_start = x_start1
        y_start = y_start1
    return None,None

class Problem2():
    def __init__(self):
      pass
    def create_dataset(self):
        RANDOM_SEED = 42
        tf.set_random_seed(RANDOM_SEED)
        n_samples = 30
        self.train_x = np.linspace(0,20,n_samples)
        self.train_y = 3.7 * self.train_x + 14 + 4 * np.random.randn(n_samples)
        plt.plot(self.train_x,self.train_y)

    def sk_solution(self):
        x = np.reshape(self.train_x,(len(self.train_x),1))
        linreg = linear_model.LinearRegression()
        linreg.fit(x,self.train_y)
        return linreg.coef_,linreg.intercept_

    def dg_solution(self,m_target,b_target):
        def dRSS_dm(m,b): 
            return(-2*sum((self.train_y-m*self.train_x-b)*self.train_x))
        def dRSS_db(m,b):
            return(-2*sum((self.train_y-m*self.train_x-b)))
        m_start = 0
        b_start = 0
        learning_rate = 0.00001 
        iterations = 2500000
        epsilon = 0.0001
        history = [(m_start,b_start)]
        for i in range(iterations):
            dW = dRSS_dm(m_start,b_start)
            db = dRSS_db(m_start,b_start)
            m_start1 = m_start - (learning_rate * dW)
            b_start1 = b_start - (learning_rate * db)
            #history.append((m_start1,b_start1))
            #print("m_start",m_start1,"m_target",m_target,abs(m_start1-m_target))
            if abs(m_start1 - m_target) <= epsilon and abs(b_start1 - b_target) <= epsilon:
                print("epsilon hit. i= ",i)
                return m_start1,b_start1
            m_start = m_start1
            b_start = b_start1
        return m_start,b_start


def main():
    print("Problem 1")
    print(problem1())
    print("Problem 2")
    problem2 = Problem2()
    problem2.create_dataset()
    m_target,b_target = problem2.sk_solution()
    print("SK",m_target,b_target)
    print("DG",problem2.dg_solution(m_target,b_target))


if __name__ == "__main__":
    main()
