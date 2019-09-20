#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os, math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

    # 읽고 쓰기 모드
    with open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def main():
    torch.manual_seed(1)

    X_train, Y_train = load_mnist('input', kind='train')
    X_test, Y_test = load_mnist('input', kind='t10k')


    X_train = torch.tensor(X_train, dtype=torch.flat)
    Y_train = torch.tensor(Y_train, dtype=torch.long)

    ndata = Y_train.shape[0]
    n_trials = 100
    eta = 0.01
    lr = 0.001

    target = torch.zeros((ndata, 10))
    for i in range(ndata):
        target[i][Y_train[i]] = 1

    W = torch.randn(X_train.shape[1], target.shape[1]) * eta
    b = torch.randn(X_train.shape[1], target.shape[1]) * eta

    # for i in range(n_trials):
        z = np.dot(X_train, W) + b
        z = nn.Softmax(z)
        print(z.shape)





if __name__ == "__main__":
    sys.exit(main())


