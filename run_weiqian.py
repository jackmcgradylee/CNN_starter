# coding=utf8
import load_mnist
import os
import numpy as np
import time
import network2
import cPickle

st = time.clock()

def li_2_softmax(labels_list):
    arr = np.zeros((len(labels_list),10))
    for idx in range(len(arr)):
        arr[idx][labels_list[idx]] = 1
    return arr

def norm_train(train):
    arr = np.zeros((len(train), 784))
    for idx,val in enumerate(train):
        for j in range(len(val)):
            if val[j]!=255:
                arr[idx][j] = val[j]/256.0
    return arr

images = np.load(u'C:/Users/Lee/Documents/手写数字识别/mnist_images.npy')
labels = np.load(u'C:/Users/Lee/Documents/手写数字识别/mnist_labels.npy')

print images
'''
data = np.hstack((images.reshape((images.shape[0], 784)), labels.reshape((images.shape[0], 1))))
np.random.seed(123456)
np.random.shuffle(data)

images_train = norm_train(data[:3000,:784])
images_test = norm_train(data[3000:,:784])
labels_train = li_2_softmax(list(data[:3000,784]))
labels_test = data[3000:,784]

images_train =  [np.reshape(images_train[i],newshape=(784,1)) for i,val in enumerate(np.zeros(len(images_train))) ]
images_test = [np.reshape(images_test[i],newshape=(784,1)) for i,val in enumerate(np.zeros(len(images_test))) ]
labels_train = [np.reshape(labels_train[i],newshape=(10,1)) for i,val in enumerate(np.zeros(len(labels_train))) ]



train_0 = zip(images_train,labels_train)
test_0 = zip(images_test,labels_test)

combind = (train_0, test_0)
cPickle.dumps(train_0,open("mnist_data.pkl","wb"))


train,cv,test = load_mnist.load_data_wrapper()

print 'Ready to train,it takes %f s to process data.'%(time.clock()-st)



net = network2.Network([784, 100, 10], cost=network2.CrossEntropyCost)

net.large_weight_initializer()
net.SGD(train_0, 40, 10, 0.1,evaluation_data=test_0, lmbda = 5.0,monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,monitor_training_cost=True, monitor_training_accuracy=True)
'''