import tensorflow as tf

def problem1():
    tensor_x = tf.constant(list(range(100,110)))
    tensor_y = tf.constant([34, 28, 45, 67, 89, 93, 24, 49, 11, 7])
    tensor_sum = tf.add(tensor_x,tensor_y)
    print("Problem #1 Lazy",tensor_sum)
    with tf.Session() as sess:
      output = sess.run(tensor_sum)
      print("Problem #1 Eager",sess.run(tensor_sum))

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
    with tf.Session() as sess:
      print("Problem #4\n",sess.run(t))
def problem5X():
    a = tf.constant(1.12,name="a")
    b = tf.constant(2.34,name="b")
    c = tf.constant(0.72,name="c")
    d = tf.constant(0.81,name="d")
    f = tf.constant(19.83,name="f")

    x = (tf.add(tf.constant(1.0),tf.add(a/b,c/(f*f))))
    s = (b-a) / (d-c)
    r = 1.0 / ((1.0/a)+(1.0/b)+(1.0/c)+(1.0/d))
    y = a*b*(1.0/c)*(f*f/2)
    return x,s,r,y


def problem5():
    x,s,r,y = problem5X()
    with tf.Session() as sess:
      print(sess.run(x))

    with tf.Session() as sess:
      print(sess.run(s))

    with tf.Session() as sess:
      print(sess.run(r))

    with tf.Session() as sess:
      print(sess.run(y))

def problem6():
    x,s,r,y = problem5X()
    with tf.Session() as sess:
        with tf.summary.FileWriter("x", sess.graph) as writer:
          sess.run(x)

    return 0
    

def main():
    problem1()
    problem2()
    problem3()
    problem3()
    problem5()
    problem6()

if __name__ == "__main__":
    main()
