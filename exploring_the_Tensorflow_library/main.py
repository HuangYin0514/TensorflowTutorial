# -*- coding: utf-8 -*-
# @Time     : 2018/11/28 19:47
# @Author   : HuangYin
# @FileName : main.py
# @Software : PyCharm

import tensorflow as tf
import numpy as np
from TestFunction import *

np.random.seed(1)

y_hat = tf.constant(36, name='y_hat')
y = tf.constant(39, name='y')
loss = tf.Variable((y - y_hat) ** 2, name='loss')

init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    print(session.run(loss))

a = tf.constant(2)
b = tf.constant(10)
c = tf.multiply(a, b)
print(c)
session = tf.Session()
print(session.run(c))

x = tf.placeholder(tf.int64, name='x')
print(session.run(2 * x, feed_dict={x: 3}))

# test linear_function
from Linear_Function import linear_function

print("result = {}".format(str(linear_function())))

print("sigmoid(0) = {}".format(str(sigmoid(0))))
print("sigmoid(12) = {}".format(str(sigmoid(12))))

logits = np.array([0.2, 0.4, 0.7, 0.9])
labels = np.array([0., 0., 1., 1.])
print("cost = {}".format(str(cost(sigmoid(logits), labels))))

labels = np.array([1, 2, 3, 0, 2, 1])
one_hot = one_hot_matrix(labels, 4)
print("one hot = \n{}".format(str(one_hot)))

# ones
print("ones = \n{}".format(str(ones([3,3]))))
