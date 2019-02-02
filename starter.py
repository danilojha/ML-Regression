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
    x = np.reshape(x, (N, np.shape(x)[1]*np.shape(x)[2]))
    print(np.shape(x))
    print(np.shape(y))
    print(np.shape(np.transpose(W)))
    #MSE loss function
    Ld = (1/(2*N))*(np.square(np.linalg.norm(np.matmul(x,W) + b - y)))
    print(Ld)
    #Weight decay loss function
    Lw = (reg/2)*np.square(np.linalg.norm(W))
    print(Lw)
    #Total Loss
    Loss = Lw + Ld
    return Loss

def gradMSE(W, b, x, y, reg):
#Gradient with respect to weights
    N = len(x)
    x = np.reshape(x, (N, np.shape(x)[1]*np.shape(x)[2]))
    a=np.matmul(x, W)
    gradW = np.linalg.norm(a + b - y)*x + reg*np.linalg.norm(W)
    print(np.shape(gradW))
    
    gradB = np.linalg.norm(a + b - y)
    print(np.shape(gradB))
    print(gradB)
    return gradW, gradB



#def crossEntropyLoss(W, b, x, y, reg):
    # Your implementation here

#def gradCE(W, b, x, y, reg):
    # Your implementation here

#def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS):
    # Your implementation here

#def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    # Your implementation here

trainData, validData, testData, trainTarget, validTarget, testTarget = loadData();
W = np.zeros((784, 1))
x = trainData
y = trainTarget
#print(W)

MSE = MSE(W, 0, x, y, 0)
print(MSE)
grad = gradMSE(W, 0, x, y, 0)

