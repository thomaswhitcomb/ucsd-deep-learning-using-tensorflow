import tensorflow as tf


def main():
    tensor_x = tf.constant(list(range(100,110)))
    tensor_y = tf.constant([34, 28, 45, 67, 89, 93, 24, 49, 11, 7])
    tensor_sum = tf.add(tensor_x,tensor_y)
    print("Problem #1 Lazy",tensor_sum)
    with tf.Session() as sess:
      output = sess.run(tensor_sum)
      print("Problem #1 Eager",sess.run(tensor_sum))

    x1 = tf.constant([[1,2,3,4],[5,6,7,8]])
    s1 = tf.stack([x1,x1,x1,x1])
    print("Problem #2",s1.shape)

    x1 = tf.constant([[1,2,3,4],[5,6,7,8]])
    s1 = tf.stack([x1])
    print("Problem #3",s1.shape)
    

if __name__ == "__main__":
    main()
