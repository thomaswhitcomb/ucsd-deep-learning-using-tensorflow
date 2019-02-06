import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import datasets
import sys

def problem1():
    RANDOM_SEED = 42
    n_hidden = 10

    #f = open("Admissions.csv")
    #f.readline()
    #dataset = np.loadtxt(fname = f, delimiter = ',')
    #features = dataset[:,1:] # antecedents
    #labels = dataset[:,:1]  # consequent
    iris = datasets.load_iris()
    features = iris["data"]
    labels = iris["target"]

    # a "1" in col 0 means not admitted
    # a "1" in col 1 means admitted
    one_hot = np.zeros(shape=(len(labels),3))
    for i in range(0,len(labels)):
        one_hot[i,int(labels[i])] = 1
    train_feats,test_feats,train_lab,test_lab = train_test_split(
            features, one_hot,test_size = 0.30 ,random_state=RANDOM_SEED)
    feat_shape = train_feats.shape[1]
    hidden_nodes = 3
    out_shape = train_lab.shape[1]

    inputs = tf.placeholder("float", shape=[None, feat_shape])
    outputs = tf.placeholder("float", shape=[None, out_shape])
    Z = tf.placeholder("float")

    W1 = tf.get_variable(name="W1",
            shape=[feat_shape, n_hidden],
            initializer=tf.contrib.layers.xavier_initializer())
    B1 = tf.get_variable(name="b1",
            shape=[n_hidden],
            initializer=tf.constant_initializer(0.0))
    H1 = tf.matmul(inputs, W1) + B1
    #H1 = tf.sigmoid(H1)
    H1 = tf.nn.relu(H1)

    W2 = tf.get_variable(name="W2",
            shape=[n_hidden, out_shape],
            initializer=tf.contrib.layers.xavier_initializer())
    B2 = tf.get_variable(name="b2",
            shape=[out_shape],
            initializer=tf.constant_initializer(0.0))
    #pred_tensor = tf.sigmoid(tf.matmul(H1, W2) + B2)
    #pred_tensor = tf.nn.relu(tf.matmul(H1, W2) + B2)
    pred_tensor = tf.matmul(H1, W2) + B2

    predict = tf.argmax(pred_tensor, axis=1)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=outputs, logits=pred_tensor))
    #cost = tf.reduce_mean(-outputs*tf.log(pred_tensor) - (1-outputs)*tf.log(1-pred_tensor))
    updates = tf.train.GradientDescentOptimizer(Z).minimize(cost)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for epoch in range(100):
            # Train with each example
            for i in range(len(train_feats)):
                op,cst = sess.run([updates,cost],
                        feed_dict={inputs: train_feats[i: i + 1], outputs: train_lab[i: i + 1],Z:0.01})
            test_accuracy = np.mean(np.argmax(test_lab, axis=1) == sess.run(predict, feed_dict={inputs: test_feats, outputs: test_lab}))
            if (epoch % 10) == 0:
                print("Epoch: %d, acc: %.2f, cost: %.5f" % (epoch, test_accuracy, cst))

def main():
    problem1()

if __name__ == "__main__":
    main()
