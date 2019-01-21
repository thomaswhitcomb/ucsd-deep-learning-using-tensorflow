import tensorflow as tf


def main():
    tensor_x = tf.constant(list(range(100,110)))
    tensor_y = tf.constant([34, 28, 45, 67, 89, 93, 24, 49, 11, 7])
    tensor_sum = tf.add(tensor_x,tensor_y)
    print(tensor_sum)
    with tf.Session() as sess:
      output = sess.run(tensor_sum)
      print(output)
    sess = tf.Session()
    print(sess.run(tensor_sum))

if __name__ == "__main__":
    main()
