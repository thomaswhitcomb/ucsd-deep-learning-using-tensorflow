import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
RANDOM_SEED = 42
n_input = 3
n_hidden = 5
n_output = 1

LR = tf.placeholder(tf.float32)
W1 = tf.Variable(tf.random_uniform([n_input,n_hidden], -1.0,1.0))
W2 = tf.Variable(tf.random_uniform([n_hidden,n_output], -1.0,1.0))
B1 = tf.Variable(tf.zeros(n_hidden),name='Bias1')
B2 = tf.Variable(tf.zeros(n_output),name='Bias2')

f = open("Admissions.csv")
f.readline()
dataset = np.loadtxt(fname = f, delimiter = ',')
features = dataset[:,1:] # antecedents
labels = dataset[:,:1]  # consequent

# a "1" in col 0 means not admitted
# a "1" in col 1 means admitted
one_hot = np.zeros(shape=(len(labels),2))
for i in range(0,len(labels)):
    one_hot[i,int(labels[i])] = 1

train_feats,tet_feats,train_lab,test_lab = train_test_split(features, one_hot,test_size =0.33 ,random_state=RANDOM_SEED)


def run(tensor):
    with tf.Session() as sess:
        return(sess.run(tensor))

def compute(inputs, weights, bias, activation_fn):
    layer = tf.matmul(inputs, weights)+bias
    return activation_fn(layer), layer

def problem1():
    print(W1)
    print(W2)
    print(B1)
    print(B2)
    print(dataset[:5,:])
    print(features[:5,:])
    print(labels[:5,:])
    print(one_hot[:5,:])
    print(train_feats.shape)

def main():
    problem1()

if __name__ == "__main__":
    main()
