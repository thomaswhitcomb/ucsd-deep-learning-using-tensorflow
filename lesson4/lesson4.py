import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import datasets
from sklearn import preprocessing
import pandas as pd
import sys

class Graph:
    def __init__(self):
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
        self.Inputs = tf.placeholder(tf.float32, shape=[None, input_shape])
        self.Outputs = tf.placeholder(tf.float32, shape=[None, out_shape])
        self.TestOutputs = tf.placeholder(tf.float32, shape=[None, out_shape])
        self.Learning_rate = tf.placeholder(tf.float32)
    
        W1 = tf.get_variable(name="W1",
                shape=[input_shape, hidden],
                initializer=self.variable_initializer()())
        B1 = tf.get_variable(name="B1",
                shape=[hidden],
                initializer=tf.constant_initializer(0.0))
        H1 = self.activation_calc()(tf.matmul(self.Inputs, W1) + B1)
        W2 = tf.get_variable(name="W2",
                shape=[hidden, out_shape],
                initializer=self.variable_initializer()())
        B2 = tf.get_variable(name="B2",
                shape=[out_shape],
                initializer=tf.constant_initializer(0.0))

        self.O1 = self.activation_calc()(tf.matmul(H1, W2) + B2)

        correct_prediction = tf.equal(tf.argmax(self.O1,axis=1), tf.argmax(self.TestOutputs,axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.cost = self.cost_calc()(self.Outputs,self.O1)
        self.updates = tf.train.GradientDescentOptimizer(self.Learning_rate).minimize(self.cost)

    def train(self,tsize,epochs,lr):
        n_axis = 1
        train_features,test_features,train_labels,test_labels = train_test_split(
             self.features, self.labels,test_size=tsize ,random_state=self.random_seed())
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            for epoch in range(epochs):
                # Train with each example
                for i in range(len(train_features)):
                    op,cst = sess.run([self.updates,self.cost], feed_dict={self.Inputs: train_features[i: i + 1], self.Outputs: train_labels[i: i + 1],self.Learning_rate:lr})
                test_accuracy = self.calc_accuracy(sess,test_features,test_labels)
                if (epoch % (epochs/10)) == 0:
                    print("Epoch: %d, acc: %.5f, cost: %.5f" % (epoch, test_accuracy, cst))

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
        return sess.run(self.accuracy,feed_dict={self.Inputs:features,self.TestOutputs:labels})

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
        xx2 = np.stack(sess.run(self.O1,feed_dict={self.Inputs:features,self.TestOutputs:labels}),axis=0)
        return sess.run(tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(xx1,xx2)))))

def problem1():
    f = open("Admissions.csv")
    f.readline()
    dataset = np.genfromtxt(fname = f, delimiter = ',')
    features = dataset[:,1:] # antecedents
    features_scaled = features/features.max(axis=0)
    labels = dataset[:,:1]  # consequent
    one_hot = np.zeros(shape=(len(labels),2))
    for i in range(0,len(labels)):
        one_hot[i,int(labels[i])] = 1

    g = Graph1()
    g.initialize(features_scaled,one_hot,4)
    g.train(0.30,500,0.01)

def problem2():
    f = open("Advertising.csv")
    f.readline()
    dataset = np.genfromtxt(fname = f, delimiter = ',')
    features = dataset[:,1:4] # antecedents
    labels = dataset[:,4:]  # consequent
    features_scaled = features/features.max(axis=0)
    labels_scaled = labels/labels.max()
    g = Graph2()
    g.initialize(features_scaled,labels_scaled,5)
    g.train(0.30,500,0.01)

def main():
    problem1()
    #problem2()

if __name__ == "__main__":
    main()
