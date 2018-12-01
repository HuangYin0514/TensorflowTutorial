# -*- coding: utf-8 -*-
# @Time     : 2018/11/29 8:20
# @Author   : HuangYin
# @FileName : Main.py
# @Software : PyCharm

from tf_utils import *
import matplotlib.pyplot as plt
import numpy as np
from Utils import *

if __name__ == '__main__':
    # load dataset
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    index = 0
    plt.imshow(X_train_orig[index])
    print("y = {}".format(str(np.squeeze(Y_train_orig[:, index]))))

    # flatten the training and test images
    X_train_flatten = X_train_orig.reshape((X_train_orig.shape[0], -1)).T
    X_test_flatten = X_test_orig.reshape((X_test_orig.shape[0], -1)).T
    # normalize image vectors
    X_train = X_train_flatten / 255
    X_test = X_test_flatten / 255
    # Convert training and test labels to one hot matrices
    Y_train = one_hot_matrix(Y_train_orig.reshape(-1), 6)
    Y_test = one_hot_matrix(Y_test_orig.reshape(-1), 6)

    print("number of training examples = " + str(X_train.shape[1]))
    print("number of test examples = " + str(X_test.shape[1]))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(Y_test.shape))

    X, Y = create_placeholders(12288, 6)
    print("X = {}".format(str(X)))
    print("Y = {}".format(str(Y)))

    tf.reset_default_graph()
    with tf.Session() as session:
        parameters = initialize_parameters()
        print("W1 = " + str(parameters["W1"]))
        print("b1 = " + str(parameters["b1"]))
        print("W2 = " + str(parameters["W2"]))
        print("b2 = " + str(parameters["b2"]))

    tf.reset_default_graph()
    with tf.Session() as session:
        X, Y = create_placeholders(12288, 6)
        parameters = initialize_parameters()
        Z3 = forward_propagation(X, parameters)
        print("Z3 = {}".format(str(Z3)))

    tf.reset_default_graph()
    with tf.Session() as session:
        X, Y = create_placeholders(12288, 6)
        parameters = initialize_parameters()
        Z3 = forward_propagation(X, parameters)
        cost = compute_cost(Z3, Y)
        print("cost = {}".format(str(cost)))

    parameters = model(X_train, Y_train, X_test, Y_test, num_epochs=2)

    import scipy
    from PIL import Image
    from scipy import ndimage

    my_image = "1.png"
    fname = "datasets/fingers/" + my_image
    image = np.array(ndimage.imread(fname, flatten=False))
    my_image = scipy.misc.imresize(image, size=(64, 64, 3)).reshape((1, 64 * 64 * 3)).T
    my_image_prediction = predict(my_image, parameters)
    plt.imshow(image)
    plt.show()
    print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))
