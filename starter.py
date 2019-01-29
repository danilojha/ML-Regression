import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def loadData():
    with np.load('notMNIST.npz') as data :
        Data, Target = data ['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

def MSE(W, b, x, y, reg):
    #tf.matmul
    #dont use tf library (only for comparison with adam)
    #W = 28*28 = 784 apparently
    #x = training or test data
    #b = 1 or 0
    #y = train or test target
    #match dimensions of matrices to multiply first
    N = len(x)
    Loss = np.sum((1/(2*N))*((np.transpose(W)*x + b - y)**2) + (reg/2)*(W))
    return Loss

#def gradMSE(W, b, x, y, reg):
    

#def crossEntropyLoss(W, b, x, y, reg):
    # Your implementation here

#def gradCE(W, b, x, y, reg):
    # Your implementation here

#def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS):
    # Your implementation here

#def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    # Your implementation here

W = np.random.rand(784, 1)
print(W)


