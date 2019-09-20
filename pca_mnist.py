#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
import numpy as np
import matplotlib as plt
from mpl_toolkits.mplot3d import Axes3D

# Get some data
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

# Subtract the mean (center data around the mean)
def meanSubtract():
    
    return

# Calculate the covariance matrix

# Calculate the eigenvectors and eigenvalues of the covariance

# Choosse componenets and form a feature vector

# Derive the new data set


def main():


if __name__ == "__main__":
    sys.exit(main())
