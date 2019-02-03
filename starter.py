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
    N = len(x)  
    #MSE loss function
    Ld = (1/(2*N))*(np.square(np.linalg.norm(np.matmul(x,W) + b - y)))
    #Weight decay loss function
    Lw = (reg/2)*np.square(np.linalg.norm(W))
    #Total Loss
    Loss = Lw + Ld
    return Loss

def gradMSE(W, b, x, y, reg):
    N = len(x)
    Wx = np.matmul(x, W)
    #gradient of MSE with respect to weights
    gradW = np.zeros((N,1))
#    gradW = (1/N)*L2*x + reg*np.linalg.norm(W)
    gradW = (1/N)*np.matmul(x.T, Wx + b - y) + reg*W
    print(np.shape(gradW))
    #gradient of MSE with respect to bias
    gradB = (1/N)*(Wx + b - y)
    return gradW, gradB

#def crossEntropyLoss(W, b, x, y, reg):
    # Your implementation here

#def gradCE(W, b, x, y, reg):
    # Your implementation here

def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS):
    x = trainingData
    y = trainingLabels
    for i in range(iterations):
        gradW, gradB = gradMSE(W, b, x, y, reg)
        print(i)
#        if np.all(gradW < EPS):
#            return W, b
        W = W - alpha*gradW
        b = b - alpha*gradB
        print(i)
    return W, b

#def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    # Your implementation here

trainData, validData, testData, trainTarget, validTarget, testTarget = loadData();
W = np.zeros((784, 1))
x = trainData
N = len(x)
x = np.reshape(x, (N, np.shape(x)[1]*np.shape(x)[2]))
y = trainTarget
print(np.shape(y))
#print(W)



mean = MSE(W, 0, x, y, 0)
print(mean)
grad = gradMSE(W, 0, x, y, 0)
W, b = grad_descent(W, 0, x, y, 0.005, 5000, 0, 0.0000001)
mean = MSE(W, 0, x, y, 0)
eff = 1 - mean
print(eff)


