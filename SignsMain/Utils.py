# -*- coding: utf-8 -*-
# @Time     : 2018/11/29 8:42
# @Author   : HuangYin
# @FileName : Utils.py
# @Software : PyCharm
import numpy as np
import tensorflow as tf

def convert_to_one_hot(labels,C):
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