###############################################
# MNIST Image Classification Using Linear Regression #
################################################
# 1.1 Load the libraries
#
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.examples.tutorials.mnist import input_data

def optimize(optimizer,num_iterations,learning_rate,batch_size):
    for i in range(num_iterations):
        x_batch, y_true_batch = data.train.next_batch(batch_size= batch_size)
        feed_dict_train = {x : x_batch,
                           lr: learning_rate,
                           y_true : y_true_batch}
        session.run(optimizer, feed_dict = feed_dict_train)

def print_confusion_matrix():
    cls_true = [np.argmax(label) for label in data.test.labels]
    cls_pred = session.run(y_pred_cls, feed_dict = feed_dict_test)
    cm = confusion_matrix(y_true = cls_true, y_pred = cls_pred)
    print(cm)

def print_accuracy(iterations,learning_rate,batch_size):
    # Use TensorFlow to compute the accuracy.
    acc = session.run(accuracy , feed_dict= feed_dict_test)
    # Print the accuracy.
    print('Accuracy : {:2.1f}% with {:d} iterations, {:1.2f} learning rate and {:d} batch size'.format((acc*100),iterations,learning_rate,batch_size))
################################################
# 1.2 Download and read MNIST data
#
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
data = input_data.read_data_sets("MNIST_data/", one_hot = True)
tf.logging.set_verbosity(old_v)
#######################################################
# the images are stored in one-dimensional arrays of this length. #
img_size_flat = data.train.images[0].shape[0]
# Tuple with height and width of images used to reshape arrays. 

img_shape = (28,28)
# Number of classes, one class for each of 10 digits. 
num_classes = 10

data.test.cls = np.array([label.argmax() for label in data.test.labels])

########################################### 
# 1.5 Plot a few images
# Get the first images from the Test-set. #
images = data.test.images[0:9]
# Get the true classes for those images.
cls_true = [np.argmax(oh) for oh in data.test.labels[0:9] ]

############################################## # 2.1 Placeholder variables
#
lr = tf.placeholder(tf.float32)
x = tf.placeholder( tf.float32, [None, img_size_flat]) 
y_true = tf.placeholder( tf.float32, [None, num_classes]) 
y_true_cls = tf.placeholder( tf.int64, [None])

############################################## # 2.2 Variables
#
weights = tf.Variable(tf.zeros([img_size_flat, num_classes])) 
bias = tf.Variable(tf.zeros([num_classes]))

############################################### # 2.3 Model
#
logits = tf.matmul(x, weights) + bias 
y_pred = tf.nn.softmax(logits) 
y_pred_cls = tf.argmax(y_pred, axis=1)

# 2.4 Cost Function
#
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2( logits= logits, labels = y_true) 
cost = tf.reduce_mean(cross_entropy)

################################################ # 2.5 Optimization Function
#
gradient_descent_optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)
adagrad_optimizer = tf.train.AdagradOptimizer(lr).minimize(cost)

# 2.6 Performance measures #
correct_prediction = tf.equal( y_pred_cls , y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

############################################## # 3.1 Create TensorFlow Session
#
session = tf.Session()

############################################# # 3.2 Initialize Variables
#

###################################################
# 3.4 Optimization Iteration
#
feed_dict_test = { 
    x : data.test.images, 
    y_true : data.test.labels, 
    y_true_cls : [np.argmax(label) for label in data.test.labels]
}

#############################################
# 4.2 Performance Iteration#1
#
# Number of iteration means how many of batchs are iterated #
print("Gradient decent optimizer")
for lrx in [x/10 for x in range(5,0,-1)]:
    session.run(tf.global_variables_initializer())
    for i in [1,9,990]:
        optimize(gradient_descent_optimizer,num_iterations= i,learning_rate = lrx,batch_size=100)
        print_accuracy(i,lrx,100)

#print_confusion_matrix()

print("Adagra optimizer ")
for lrx in [x/10 for x in range(5,0,-1)]:
    session.run(tf.global_variables_initializer())
    for i in [1,9,990]:
        optimize(adagrad_optimizer,num_iterations= i,learning_rate = lrx,batch_size=100)
        print_accuracy(i,lrx,100)

#print_confusion_matrix()
print("Adagra optimizer with incremental batch size ")
session.run(tf.global_variables_initializer())
for lrx in [x/10 for x in range(5,0,-1)]:
    for b in range(1,1000,100):
        session.run(tf.global_variables_initializer())
        for i in [1,9,990]:
            optimize(adagrad_optimizer,num_iterations= i,learning_rate = lrx,batch_size=100)
            print_accuracy(i,lrx,b)

print_confusion_matrix()
