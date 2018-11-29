# -*- coding: utf-8 -*-
# @Time     : 2018/11/28 21:19
# @Author   : HuangYin
# @FileName : TestFunction.py
# @Software : PyCharm

import tensorflow  as tf


def sigmoid(z):
    """
       Computes the sigmoid of z

       Arguments:
       z -- input value, scalar or vector

       Returns:
       results -- the sigmoid of z
       """
    x = tf.placeholder(tf.float32, name='x')
    sigmoid = tf.sigmoid(x)

    with tf.Session() as session:
        result = session.run(sigmoid, feed_dict={x: z})

    return result


def cost(logits, lables):
    """
    Computes the cost using the sigmoid cross entropy

    Arguments:
    logits -- vector containing z, output of the last linear unit (before the final sigmoid activation)
    labels -- vector of labels y (1 or 0)

    Note: What we've been calling "z" and "y" in this class are respectively called "logits" and "labels"
    in the TensorFlow documentation. So logits will feed into z, and labels into y.

    Returns:
    cost -- runs the session of the cost (formula (2))
    """
    z = tf.placeholder(tf.float32, name="z")
    y = tf.placeholder(tf.float32, name="y")
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y)
    with tf.Session() as session:
        result = session.run(cost, feed_dict={z: logits, y: lables})
    return result


def one_hot_matrix(labels, C):
    """
      Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                       corresponds to the jth training example. So if example j had a label i. Then entry (i,j)
                       will be 1.

      Arguments:
      labels -- vector containing the labels
      C -- number of classes, the depth of the one hot dimension

      Returns:
      one_hot -- one hot matrix
      """
    C = tf.constant(C, name="C")
    # features x depth if axis == -1
    # depth x features if axis == 0
    one_hot_matrix = tf.one_hot(indices=labels, depth=C, axis=0)

    with tf.Session() as session:
        one_hot = session.run(one_hot_matrix)

    return one_hot


def ones(shape):
    """
        Creates an array of ones of dimension shape

        Arguments:
        shape -- shape of the array you want to create

        Returns:
        ones -- array containing only ones
        """
    ones = tf.ones(shape)
    session = tf.Session()
    ones = session.run(ones)
    session.close()
    return ones
