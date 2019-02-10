import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import datasets
import sys

def cost1(target,compValue):
    return tf.reduce_mean(-target*tf.log(compValue) - (1-target)*tf.log(1-compValue))

def cost2(target,compValue):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=target, logits=compValue))

def cost3(target,compValue):
    return tf.reduce_mean(tf.pow(target - compValue,2))

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
        

def train2(features,labels,p_hidden,p_cost_fn=cost1,p_activation_fn=tf.nn.relu,p_epochs=100,p_learning_rate=0.1,p_test_size=0.0,p_initializer=tf.initializers.random_uniform):
    RANDOM_SEED = 42
    tf.set_random_seed(RANDOM_SEED)

    input_shape = features.shape[1]
    out_shape = labels.shape[1]

    Inputs = tf.placeholder(tf.float32, shape=[None, input_shape])
    Outputs = tf.placeholder(tf.float32, shape=[None, out_shape])
    Y = tf.placeholder(tf.float32, shape=[None, out_shape])
    Learning_rate = tf.placeholder(tf.float32)

    W1 = tf.get_variable(name="W1",
            shape=[input_shape, p_hidden],
            initializer=p_initializer())
    B1 = tf.get_variable(name="B1",
            shape=[p_hidden],
            initializer=tf.constant_initializer(0.0))
    H1 = p_activation_fn(tf.matmul(Inputs, W1) + B1)

    W2 = tf.get_variable(name="W2",
            shape=[p_hidden, out_shape],
            initializer=p_initializer())
    B2 = tf.get_variable(name="B2",
            shape=[out_shape],
            initializer=tf.constant_initializer(0.0))

    O1wo = tf.matmul(H1, W2) + B2
    O1 = p_activation_fn(O1wo)
    #O1 = tf.matmul(H1, W2) + B2
    n_axis = 1
    predict = tf.argmax(O1, axis=n_axis)
    predict2 = O1
    correct_prediction = tf.equal(tf.argmax(O1,axis=n_axis), tf.argmax(Y,axis=n_axis))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    cost = p_cost_fn(Outputs,O1)
    updates = tf.train.GradientDescentOptimizer(Learning_rate).minimize(cost)

    train_features,test_features,train_labels,test_labels = train_test_split(
            features, labels,test_size=p_test_size ,random_state=RANDOM_SEED)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for epoch in range(p_epochs):
            # Train with each example
            for i in range(len(train_features)):
                op,cst = sess.run([updates,cost], feed_dict={Inputs: train_features[i: i + 1], Outputs: train_labels[i: i + 1],Learning_rate:p_learning_rate})
            #test_accuracy = np.mean(np.argmax(test_labels, axis=n_axis) == sess.run(predict, feed_dict={Inputs: test_features, Outputs: test_labels}))

            #test_accuracy1 = sess.run(accuracy,feed_dict={Inputs:test_features,Y:test_labels})
            if (epoch % (p_epochs/10)) == 0:
                xx1 = np.stack(test_labels,axis=0)
                xx2 = np.stack(sess.run(O1,feed_dict={Inputs:test_features,Y:test_labels}),axis=0)
                test_accuracy = sess.run(tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(xx1,xx2)))))
                print("Epoch: %d, acc: %.5f, cost: %.5f" % (epoch, test_accuracy, cst))

def verify1():
    f = open("Xor.csv")
    f.readline()
    dataset = np.loadtxt(fname = f, delimiter = ',')
    features = dataset[:,1:] # antecedents
    labels = dataset[:,:1]  # consequent

    train(
            features,
            labels,
            3,
            p_epochs=10000,
            p_cost_fn=cost1,
            p_activation_fn=tf.sigmoid,
            p_learning_rate=0.1,
            p_initializer=tf.initializers.random_uniform,
            p_test_size=0.0)    

def verify2():
    iris = datasets.load_iris()
    features = iris["data"]
    labels = iris["target"]

    one_hot = np.zeros(shape=(len(labels),3))
    for i in range(0,len(labels)):
        one_hot[i,labels[i]] = 1
    print("one hot: ",one_hot[:5,:])
    train(
            features,
            one_hot,
            10,
            p_epochs=100,
            p_cost_fn=cost2,
            p_activation_fn=tf.nn.relu,
            p_learning_rate=0.01,
            p_initializer=tf.contrib.layers.xavier_initializer,
            p_test_size=0.33)    

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
    g.initialize(features_scaled,one_hot,10)
    g.train(0.30,500,0.001)

def problem2():
    f = open("Advertising.csv")
    f.readline()
    dataset = np.genfromtxt(fname = f, delimiter = ',')
    features = dataset[:,1:4] # antecedents
    print("features.shape:",features.shape)
    labels = dataset[:,4:]  # consequent
    print("labels.shape:",labels.shape)
    features_scaled = features/features.max(axis=0)
    print("features_scaled.shape:",features_scaled.shape)
    labels_scaled = labels/labels.max()
    print("labels_scaled.shape:",labels_scaled.shape)
    train2(
            features_scaled,
            labels_scaled,
            5,
            p_epochs=5000,
            p_cost_fn=cost3,
            p_activation_fn=tf.nn.relu,
            p_learning_rate=0.01,
            #p_initializer=tf.contrib.layers.xavier_initializer,
            p_initializer=tf.initializers.random_uniform,
            p_test_size=0.33)    

def main():
    #verify1()
    #verify2()
    problem1()
    #problem2()
    #problem1X()

if __name__ == "__main__":
    main()
