import tensorflow as tf
import sys

def run(tensor):
    with tf.Session() as sess:
        return(sess.run(tensor))

# Y = X activation function
def y_equal_x(x):
    return x

def compute(inputs,weights,bias,activation_fn):
    layer = tf.matmul(inputs,weights)+bias
    return activation_fn(layer),layer


def problem1():
    inputs = tf.constant([[100,150]])

    layer1Weights = tf.constant([[5,10,15,20],[25,30,35,40]])
    layer1Bias = tf.constant([[17,19,21,23]])
    layer2Weights = tf.constant([[30,35],[40,45],[70,75],[80,85]])
    layer2Bias = tf.constant([[35,36]])

    activation,l1 = compute(inputs,layer1Weights,layer1Bias,y_equal_x)
    activation,l2 = compute(activation,layer2Weights,layer2Bias,y_equal_x)
    return run(l2)

def problem2():
    inputs = tf.constant([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
    weights = tf.constant([[-4.,-6.,-5.],[3.,6.,4.]])
    bias = tf.constant([[-2.,3.,-2.]])
    activation,l = compute(inputs,weights,bias,tf.sigmoid)
  
    weights = tf.constant([[5.],[-9.],[7.]])
    bias = tf.constant([4.])
    activation,l = compute(activation,weights,bias,tf.sigmoid)
    return run(activation)
   
def main():
    print(problem1())
    print(problem2())

if __name__ == "__main__":
    main()
