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


def train(features,labels,p_hidden,p_cost_fn=cost1,p_activation_fn=tf.nn.relu,p_epochs=100,p_learning_rate=0.1,p_test_size=0.0,p_initializer=tf.initializers.random_uniform):
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
    correct_prediction = tf.equal(tf.argmax(O1,axis=n_axis), tf.argmax(Y,axis=n_axis))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    cost = p_cost_fn(Outputs,O1)
    updates = tf.train.GradientDescentOptimizer(Learning_rate).minimize(cost)

    train_features,test_features,train_labels,test_labels = train_test_split(
            features, labels,test_size=p_test_size ,random_state=RANDOM_SEED)
    #print("number of training features is",len(train_features))
    #print("number of test features is",len(test_features))
    #print("train_features shape:",train_features.shape)
    #print("train_labels shape:",train_labels.shape)
    #print("train_features:",train_features[:5,:])
    #print("train_labels:",train_labels[:5,:])
    #print("test_labels:",test_labels.shape,test_labels[:5,:])
    #print("np.argmax(test_labels,axis=1):",np.argmax(test_labels))
    #print("test_labels[np.argmax(test_labels,axis=1)]:",test_labels[np.argmax(test_labels)])
    #print("np.mean(np.argmax(test_labels,axis=1)):",np.mean(np.argmax(test_labels)))
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for epoch in range(p_epochs):
            # Train with each example
            for i in range(len(train_features)):
                op,cst = sess.run([updates,cost], feed_dict={Inputs: train_features[i: i + 1], Outputs: train_labels[i: i + 1],Learning_rate:p_learning_rate})
            test_accuracy = np.mean(np.argmax(test_labels, axis=n_axis) == sess.run(predict, feed_dict={Inputs: test_features, Outputs: test_labels}))
            test_accuracy1 = sess.run(accuracy,feed_dict={Inputs:test_features,Y:test_labels})
            if (epoch % (p_epochs/10)) == 0:
                x1 = np.argmax(test_labels, axis=n_axis)
                x2 = sess.run(predict, feed_dict={Inputs: test_features, Outputs: test_labels})
                x3 = np.equal(x1,x2)
                x4 = np.mean(x3)
                print("observed labels:", test_labels)
                print("generated labels:", x2)
                print("x1:", x1)
                print("x2:", x2)
                print("x3:", x3)
                print("x4:", x4)
                print("Epoch: %d, acc: %.5f,acc: %.5f, cost: %.5f" % (epoch, test_accuracy,test_accuracy1, cst))

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
    dataset = np.loadtxt(fname = f, delimiter = ',')
    features = dataset[:,1:] # antecedents
    features_scaled = features/features.max(axis=0)
    labels = dataset[:,:1]  # consequent

    train(
            features_scaled,
            labels,
            10,
            p_epochs=100,
            p_cost_fn=cost1,
            p_activation_fn=tf.sigmoid,
            p_learning_rate=0.1,
            p_initializer=tf.initializers.random_uniform,
            p_test_size=0.30)    

def problem2():
    f = open("Advertising.csv")
    f.readline()
    dataset = np.genfromtxt(fname = f, delimiter = ',')
    features = dataset[:,1:4] # antecedents
    labels = dataset[:,4:]  # consequent
    features_scaled = features/features.max(axis=0)
    labels_scaled = labels/labels.max()
    train(
            features_scaled,
            labels_scaled,
            10,
            p_epochs=100,
            p_cost_fn=cost3,
            #p_activation_fn=tf.sigmoid,
            p_activation_fn=tf.nn.relu,
            p_learning_rate=0.001,
            #p_initializer=tf.contrib.layers.xavier_initializer,
            p_initializer=tf.initializers.random_uniform,
            p_test_size=0.33)    

def main():
    #verify1()
    #verify2()
    #problem1()
    problem2()

if __name__ == "__main__":
    main()
