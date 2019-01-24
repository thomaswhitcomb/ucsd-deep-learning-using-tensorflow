import tensorflow as tf

class Tensors:
    def __init__(self):
        tf.reset_default_graph()
        self.a = tf.constant(1.12,name="a")
        self.b = tf.constant(2.34,name="b")
        self.c = tf.constant(0.72,name="c")
        self.d = tf.constant(0.81,name="d")
        self.f = tf.constant(19.83,name="f")

    def x(self):
         return (tf.add(tf.constant(1.0),tf.add(tf.divide(self.a,self.b),tf.divide(self.c,(tf.square(self.f))))))

    def s(self):
        return tf.divide(tf.subtract(self.b,self.a) , tf.subtract(self.d,self.c))

    def r(self):
        return tf.divide(1.0 , (tf.divide(1.0,self.a)+tf.divide(1.0,self.b)+tf.divide(1.0,self.c)+tf.divide(1.0,self.d)))

    def y(self):
        return tf.multiply( self.a, self.b)*tf.divide(1.0,self.c)*tf.multiply(self.f,tf.divide(self.f,2))

    def test(self):
        if run(self.x()).item() != 1.4804635047912598:
          print("Tensors - x is bad: ",run(self.x()).item())
        if run(self.s()).item() != 13.555558204650879:
          print("Tensors - s is bad: ",run(self.s()).item())
        if run(self.r()).item() != 0.2535712718963623:
          print("Tensors - r is bad: ",run(self.r()).item())
        if run(self.y()).item() != 715.676513671875:
          print("Tensors - y s bad: ",run(self.y()).item())
      

def problem1():
    tensor_x = tf.constant(list(range(100,110)))
    tensor_y = tf.constant([34, 28, 45, 67, 89, 93, 24, 49, 11, 7])
    tensor_sum = tf.add(tensor_x,tensor_y)
    print("Problem #1 Lazy",tensor_sum)
    print("Problem #1 Eager",run(tensor_sum))

def problem2():
    x = tf.constant([[1,2,3,4],[5,6,7,8]])
    t = tf.stack([x,x,x,x])
    print("Problem #2",t.shape)

def problem3():
    x = tf.constant([[1,2,3,4],[5,6,7,8]])
    t = tf.stack([x])
    print("Problem #3",t.shape)

def problem4():
    x = tf.constant([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
    t = tf.reshape(x,[6,2])
    print("Problem #4",t.shape)
    print("Problem #4\n",run(t))

def run(tensor):
    with tf.Session() as sess:
        return(sess.run(tensor))

def problem5():
    tensors = Tensors()
    tensors.test()
    print("Problem #5")
    print("x = ",run(tensors.x()))
    print("s = ",run(tensors.s()))
    print("r = ",run(tensors.r()))
    print("y = ",run(tensors.y()))

def graphit(name,tensor):
    with tf.Session() as sess:
        with tf.summary.FileWriter(name, sess.graph) as writer:
             sess.run([tensor])

def problem6():
      
    graphit("graphs/x",Tensors().x())
    graphit("graphs/s",Tensors().s())
    graphit("graphs/r",Tensors().r())
    graphit("graphs/y",Tensors().y())

def main():
    problem1()
    problem2()
    problem3()
    problem4()
    problem5()
    problem6()

if __name__ == "__main__":
    main()
