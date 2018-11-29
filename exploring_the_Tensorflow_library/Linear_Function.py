# -*- coding: utf-8 -*-
# @Time     : 2018/11/28 21:08
# @Author   : HuangYin
# @FileName : Linear_Function.py
# @Software : PyCharm

import numpy as np
import tensorflow as tf


def linear_function():
    """
     Implements a linear function:
             Initializes W to be a random tensor of shape (4,3)
             Initializes X to be a random tensor of shape (3,1)
             Initializes b to be a random tensor of shape (4,1)
     Returns:
     result -- runs the session for Y = WX + b
     """
    np.random.seed(1)

    X = np.random.randn(3, 1)
    W = np.random.randn(4, 3)
    b = np.random.randn(4, 1)
    Y = tf.add(tf.matmul(W, X), b)

    with tf.Session() as session:
        result = session.run(Y)

    return result
