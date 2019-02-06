import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import datasets
import sys

def cost1(target,compValue):
    x = tf.reduce_mean(-target*tf.log(compValue) - (1-target)*tf.log(1-compValue))
    return x

def cost2(target,compValue):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=target, logits=compValue))

def compute(features,labels,n_hidden,cost_fn=cost1,activation_fn=tf.nn.relu,epochs=100,learning_rate=0.1,n_test_size=0.0,n_initializer=tf.initializers.random_uniform):
    RANDOM_SEED = 42

    input_shape = features.shape[1]
    out_shape = labels.shape[1]

    inputs = tf.placeholder("float", shape=[None, input_shape])
    outputs = tf.placeholder("float", shape=[None, out_shape])

    Z = tf.placeholder("float")

    W1 = tf.get_variable(name="W1",
            shape=[input_shape, n_hidden],
            initializer=n_initializer())
    B1 = tf.get_variable(name="b1",
            shape=[n_hidden],
            initializer=tf.constant_initializer(0.0))
    H1 = activation_fn(tf.matmul(inputs, W1) + B1)

    W2 = tf.get_variable(name="W2",
            shape=[n_hidden, out_shape],
            initializer=n_initializer())
    B2 = tf.get_variable(name="b2",
            shape=[out_shape],
            initializer=tf.constant_initializer(0.0))

    pred_tensor = activation_fn(tf.matmul(H1, W2) + B2)

    predict = tf.argmax(pred_tensor, axis=1)
    cost = cost_fn(outputs,pred_tensor)
    updates = tf.train.GradientDescentOptimizer(Z).minimize(cost)

    train_features,test_features,train_labels,test_labels = train_test_split(
            features, labels,test_size=n_test_size ,random_state=RANDOM_SEED)
    print("number of training features is",len(train_features))
    print("number of test features is",len(test_features))

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for epoch in range(epochs):
            # Train with each example
            for i in range(len(train_features)):
                op,cst = sess.run([updates,cost],
                        feed_dict={inputs: train_features[i: i + 1], outputs: train_labels[i: i + 1],Z:learning_rate})
            if len(test_features) >0:    
                test_accuracy = np.mean(np.argmax(test_labels, axis=1) == sess.run(predict, feed_dict={inputs: test_features, outputs: test_labels}))
            else:
                test_accuracy = 0.0

            if (epoch % (epochs/10)) == 0:
                print("Epoch: %d, acc: %.2f, cost: %.5f" % (epoch, test_accuracy, cst))

def problem1():
    f = open("Xor.csv")
    f.readline()
    dataset = np.loadtxt(fname = f, delimiter = ',')
    features = dataset[:,1:] # antecedents
    labels = dataset[:,:1]  # consequent

    # a "1" in col 0 means not admitted
    # a "1" in col 1 means admitted
    one_hot = np.zeros(shape=(len(labels),3))
    for i in range(0,len(labels)):
        one_hot[i,int(labels[i])] = 1

    compute(
            features,
            labels,
            3,
            epochs=10000,
            cost_fn=cost1,
            activation_fn=tf.sigmoid,
            learning_rate=0.1,
            n_test_size=0.00)    

def problem2():
    iris = datasets.load_iris()
    features = iris["data"]
    labels = iris["target"]

    # a "1" in col 0 means not admitted
    # a "1" in col 1 means admitted
    one_hot = np.zeros(shape=(len(labels),3))
    for i in range(0,len(labels)):
        one_hot[i,int(labels[i])] = 1

    compute(
            features,
            one_hot,
            10,
            epochs=100,
            cost_fn=cost2,
            activation_fn=tf.nn.relu,
            learning_rate=0.01,
            n_test_size=0.33)    

def main():
    problem2()

if __name__ == "__main__":
    main()
