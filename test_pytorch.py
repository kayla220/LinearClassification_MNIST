#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


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


class LinearMnist(nn.Module):
    def __init__(self):
        super(LinearMnist, self).__init__()

        X_train, Y_train = load_mnist('input', kind='train')
        X_test, Y_test = load_mnist('input', kind='t10k')

        self.linear1 = nn.Linear(784, 10)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.softmax(x)
        return x


def main():
    X_train, Y_train = load_mnist('input', kind='train')
    X_test, Y_test = load_mnist('input', kind='t10k')

    X_train = torch.tensor(X_train, dtype=torch.float)
    Y_train = torch.tensor(Y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float)
    Y_test = torch.tensor(Y_test, dtype=torch.long)

    model = LinearMnist()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0008)
    print(model)

    for i in range(1000):
        out = model(X_train)
        loss = loss_function(out, Y_train)
        loss.backward()
        optimizer.step()
        print(i, loss.item())

        if i % 10 == 0:
            out = model(X_test)
            out = torch.argmax(out, dim=1)
            print('acc', (out==Y_test).sum().item()/ Y_test.size(0))

    out = model(X_test)
    out = torch.argmax(out, dim=1)
    print('acc', (out==Y_test).sum().item()/ Y_test.size(0))



if __name__ == "__main__":
    sys.exit(main())

