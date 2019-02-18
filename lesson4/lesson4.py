import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import datasets
from sklearn import preprocessing
import pandas as pd
import sys

class Graph:
    def __init__(self):
        tf.reset_default_graph()
        tf.set_random_seed(self.random_seed())

    def cost_calc(self):
        print("Need to override calc_cost")

    def activation_calc(self):
        print("Need activaation calc")

    def variable_initializer(self):
        print("Missing variable_initializer")

    def calc_accuracy(self,sess,features,labels):
        print("Missing calc_accuracy")

    def random_seed(self):
        return 42

    def initialize(self,features,labels,hidden):
        self.features = features
        self.labels = labels
        self.hidden = hidden

        input_shape = self.features.shape[1]
        out_shape = self.labels.shape[1]
        self.inputs = tf.placeholder(tf.float32, shape=[None, input_shape])
        self.output = tf.placeholder(tf.float32, shape=[None, out_shape])
        self.test_outputss = tf.placeholder(tf.float32, shape=[None, out_shape])
        self.learning_rate = tf.placeholder(tf.float32)
    
        self.w1 = tf.get_variable(name="w1",
                shape=[input_shape, hidden],
                initializer=self.variable_initializer()())
        self.b1 = tf.get_variable(name="b1",
                shape=[hidden],
                initializer=tf.constant_initializer(0.0))
        self.h1 = self.activation_calc()(tf.matmul(self.inputs, self.w1) + self.b1)
        self.w2 = tf.get_variable(name="w2",
                shape=[hidden, out_shape],
                initializer=self.variable_initializer()())
        self.b2 = tf.get_variable(name="b2",
                shape=[out_shape],
                initializer=tf.constant_initializer(0.0))

        self.o1 = self.activation_calc()(tf.matmul(self.h1, self.w2) + self.b2)

        correct_prediction = tf.equal(tf.argmax(self.o1,axis=1), tf.argmax(self.test_outputss,axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.cost = self.cost_calc()(self.output,self.o1)
        self.updates = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

    def train(self,tsize,epochs,lr):
        train_features,test_features,train_labels,test_labels = train_test_split(
             self.features, self.labels,test_size=tsize ,random_state=self.random_seed())
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            for epoch in range(epochs+1):
                # Train with each example
                for i in range(len(train_features)):
                    op,cst = sess.run([self.updates,self.cost], feed_dict={self.inputs: train_features[i: i + 1], self.output: train_labels[i: i + 1],self.learning_rate:lr})
                if (epoch % (epochs/20)) == 0:
                    test_accuracy = self.calc_accuracy(sess,test_features,test_labels)
                    print("Epoch: %d, accuracy: %.5f, cost: %.5f" % (epoch, test_accuracy, cst))

            print("Weights Level 1:\n",sess.run(self.w1))
            print("Bias Level 1:\n",sess.run(self.b1))
            print("Weights Level 2:\n",sess.run(self.w2))
            print("Bias Level 2:\n",sess.run(self.b2))
 
            return test_accuracy,cst

class Graph1(Graph):
    def cost_calc(self):
        def calculator(target,compValue):
            return tf.reduce_mean(-target*tf.log(compValue) - (1-target)*tf.log(1-compValue))
        return calculator
    def activation_calc(self):
        return tf.sigmoid
    def variable_initializer(self):
        return tf.contrib.layers.xavier_initializer
    def calc_accuracy(self,sess,features,labels):
        return sess.run(self.accuracy,feed_dict={self.inputs:features,self.test_outputss:labels})

class Graph2(Graph):
    def cost_calc(self):
        def calculator(target,compValue):
            return tf.reduce_mean(tf.pow(target - compValue,2))
        return calculator

    def activation_calc(self):
        return tf.nn.relu

    def variable_initializer(self):
        return tf.initializers.random_uniform

    def calc_accuracy(self,sess,features,labels):
        xx1 = np.stack(labels,axis=0)
        xx2 = np.stack(sess.run(self.o1,feed_dict={self.inputs:features,self.test_outputss:labels}),axis=0)
        return sess.run(tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(xx1,xx2)))))

def scale(t):
    tMin = t.min(axis=0)
    tMax = t.max(axis=0)
    return (t-tMin)/(tMax-tMin)

def problem1(epochs,learning_rate,hidden):
    f = open("Admissions.csv")
    f.readline()
    dataset = np.genfromtxt(fname = f, delimiter = ',')
    features = dataset[:,1:] # antecedents
    features_scaled = scale(features)
    labels = dataset[:,0:1]  # consequent
    one_hot = np.zeros(shape=(len(labels),2))
    for i in range(0,len(labels)):
        one_hot[i,int(labels[i])] = 1

    g = Graph1()
    g.initialize(features_scaled,one_hot,hidden)
    return g.train(0.30,epochs,learning_rate)

def problem2(epochs,learning_rate,hidden):
    f = open("Advertising.csv")
    f.readline()
    dataset = np.genfromtxt(fname = f, delimiter = ',')
    features = dataset[:,1:4] # antecedents
    features_scaled = scale(features)
    labels = dataset[:,4:]  # consequent
    labels_scaled = scale(labels)
    g = Graph2()
    g.initialize(features_scaled,labels_scaled,hidden)
    return g.train(0.30,epochs,learning_rate)

def main():
    print("Problem 1")
    print("=========")
    problem1(375,0.1,9)
    print("Problem 2")
    print("=========")
    print("Accuracy is via RMSE")
    problem2(500,0.01,5)

if __name__ == "__main__":
    main()
