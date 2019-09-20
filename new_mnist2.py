5#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os 
import numpy as np
import matplotlib.pyplot as plt


def load_mnist(path, kind='train'):

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,'%s-labels-idx1-ubyte'% kind)
    images_path = os.path.join(path,'%s-images-idx3-ubyte'% kind)
    
    # 읽고 쓰기 모드 
    with open(labels_path,'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)
    
    with open(images_path,'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


# Forward Propagation
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def ReLU(z):
    return np.maximum(0.0, z)


def softmax(A):
    c = np.max(A)
    expA = np.exp(A - c)
    return expA / np.sum(expA, axis=1).reshape(A.shape[0], 1)
# Cost Function = loss function = 작은값이 나오면 좋ㅇ
# Y: target(정답)  Y_hat: output (마지막층의 출력)
# 분포의 차이를 비교하기 위해서 if, 거리를 비교 --> Euclidean을 썼겠지..
# 값이 클수록 불확실성이 커진다.


def cost_func(Y, Y_hat):
    m = Y.shape[0]
    #return -np.sum(np.multiply(Y, np.log(Y_hat))) / m
    return -np.sum(np.multiply(Y, np.log(Y_hat.clip(min=1e-10)))) / m


# BackProp      
def delta_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))


def delta_loss(Y, Y_hat):
    return Y_hat - Y


def main():
    ## Load the data
    # X_train, Y_train (60000,784):training images (60000,):label
    
    X_train, Y_train = load_mnist('input', kind='train')
    X_test, Y_test = load_mnist('input', kind='t10k')

    target = np.zeros((Y_train.shape[0], 10))
    for i in range (Y_train.shape[0]):
        target[i][Y_train[i]] = 1

    np.random.seed(0)
    lr = 0.00000001
    W = np.random.randn(X_train.shape[1], target.shape[1]) * 0.01
    b = np.random.randn(1,target.shape[1]) * 0.01
  
    #avg_target = np.average(target, axis=0)
    for i in range(100):
        z = np.dot(X_train, W) + b
        z = softmax(z)
        loss = cost_func(target, z)
        print(i, loss)

        #avg_z = np.average(z, axis=0)
        #dz = (avg_z - avg_target).reshape(1, z.shape[1])
        dz = z - target
        b -= lr * (np.average(dz, axis=0))
        #W -= lr * (np.dot(dz.T, X_train)).T
        W -= lr * (np.dot(X_train.T, dz))

    z = np.dot(X_test, W) + b
    z = softmax(z)
    z = np.argmax(z, axis=1)

    print('Accuracy: ', 1 - np.count_nonzero(z-Y_test)/Y_test.shape[0])


if __name__ == "__main__":
    sys.exit(main())
