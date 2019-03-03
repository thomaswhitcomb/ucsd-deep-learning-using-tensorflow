###############################################
# MNIST Image Classification Using Linear Regression #
################################################
# 1.1 Load the libraries
#
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.examples.tutorials.mnist import input_data
print(tf.__version__)

def plot_images(images, cls_true , cls_pred=None): 
    assert len(images) == len(cls_true) == 9
    fig, axes = plt.subplots(3,3) 
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
        # Show true and predicted classes 
        if cls_pred is None:
            xlabel = 'True : {0}'.format(cls_true[i]) 
        else: 
            xlabel = 'True : {0}, Pred : {1}'.format(cls_true[i], cls_pred[i]) 
        ax.set_xlabel(xlabel)
        # Remove ticks from the plot
        ax.set_xticks([])
        ax.set_yticks([])
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell. 
    plt.show()

###########################################
# 3.3 Function to perform optimization iteration #
batch_size = 100
def optimize(num_iterations): 
    for i in range(num_iterations):
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = data.train.next_batch(batch_size= batch_size)
        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph. # Note that the placeholder for y_true_cls is not set # because it is not used during training. 
        feed_dict_train = {x : x_batch,
                           y_true : y_true_batch}
        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict train
        # to the placeholder variables and then runs the optimizer. 
        session.run(optimizer, feed_dict = feed_dict_train)

def print_confusion_matrix():
    # Get the true classifications for the Test-set.
    cls_true = [np.argmax(label) for label in data.test.labels]
    # Get the predicted classifications for the Test-set.
    cls_pred = session.run(y_pred_cls, feed_dict = feed_dict_test)
    # Get the confusion matrix using sklearn. cm = confusion_matrix(y_true = cls_true, y_pred = cls_pred)
    # Print the confusion matrix as text. print(cm)
    # Plot the confusion matrix as an image.
    plt.imshow(cm, interpolation ='nearest', cmap = plt.cm.Blues)
    # Make various adjustments to the plot. plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes) 
    plt.xticks(tick_marks, range(num_classes)) 
    plt.yticks(tick_marks, range(num_classes)) 
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # Ensure the plot is shown correctly with multiple plots # in a single Notebook cell.
    plt.show()        

def plot_example_errors():
    correct, cls_pred = session.run([correct_prediction, 
                                     y_pred_cls], 
                                    feed_dict = feed_dict_test)
    # Negate the boolean array. incorrect = (correct == False)
    # Get the images from the Test-set that have been # incorrectly classified
    images = data.test.images[incorrect]
    # Get the predicted classes for those images 
    cls_pred = cls_pred[incorrect]
    # Get the true classes for those images.
    cls_true = [np.argmax(label) for label in data.test.labels[incorrect]]
    # Plot the first 9 images. 
    plot_images(images = images[0:9], 
                cls_true = cls_true[0:9], 
                cls_pred = cls_pred[0:9])

def plot_weights():
    # Get the values for the weights from the TensforFlow variable 
    w = session.run(weights)
    w_min = np.min(w)
    w_max = np.max(w)
    # Create figure with 3x4 sub-plots,
    # where the last 2 sub-plots are unused. 
    fig, axes = plt.subplots(3,4) 
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        # Only use the weights for the first 10 sub-plots 
        if i < 10:
            # Get the weights for the i'th digit and reshape it. 
            # Note that w.shape == (img_size_flat, 10)
            image = w[:, i].reshape(img_shape)
            # Set the label for the sub-plot. 
            ax.set_xlabel('Weights : {0}'.format(i))
            # Plot the image.
            ax.imshow(image, vmin=w_min, vmax= w_max, cmap = 'seismic')
        # Remove ticks from each sub-plot
        ax.set_xticks([])
        ax.set_yticks([])
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()   

def print_accuracy():
    # Use TensorFlow to compute the accuracy.
    acc = session.run(accuracy , feed_dict= feed_dict_test)
    # Print the accuracy.
    print('Accuracy on Test-st : {0:.1%}'.format(acc))
################################################
# 1.2 Download and read MNIST data
#
data = input_data.read_data_sets("MNIST_data/", one_hot = True)

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
# Plot the images and labels using our helper-function above. 
plot_images(images, cls_true)

############################################## # 2.1 Placeholder variables
#
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
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

# 2.6 Performance measures #
correct_prediction = tf.equal( y_pred_cls , y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

############################################## # 3.1 Create TensorFlow Session
#
session = tf.Session()

############################################# # 3.2 Initialize Variables
#
session.run(tf.global_variables_initializer())

###################################################
# 3.4 Optimization Iteration
#
feed_dict_test = { 
    x : data.test.images, 
    y_true : data.test.labels, 
    y_true_cls : [np.argmax(label) for label in data.test.labels]
}
print_accuracy()
plot_example_errors()

#############################################
# 4.2 Performance Iteration#1
#
# Number of iteration means how many of batchs are iterated #
optimize(num_iterations= 1)
print_accuracy()
plot_example_errors()
############################################### 
# 4.3 Performance Iteration #10
# We've already performed 1 iteration.
#
optimize( num_iterations=9 )
print_accuracy()
plot_example_errors()

################################################# 
# 4.4 Performance Iteration #1000
# We've already performed 10 iterations.
#
optimize(num_iterations=990)
print_accuracy()
plot_example_errors()

################################################### 
# 4.5 Print confusion Matrix
#
print_confusion_matrix()
