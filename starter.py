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
    gradW = (1/N)*np.matmul(x.T, Wx + b - y) + reg*W
    #gradient of MSE with respect to bias
    gradB = np.dot(np.ones((1, x.shape[0])), Wx + b - y)*(1/N)
    return gradW, gradB

def crossEntropyLoss(W, b, x, y, reg):
    #Binary cross entropy loss
    N = len(x)
    yhat = 1/(1 + np.exp(-(W.T*x + b)))
    Ld = (1/N)*(np.square(-y*np.log(yhat) - ((1 - y)*np.log(1 - yhat))))
    
    #Weight decay loss function
    Lw = (reg/2)*np.square(np.linalg.norm(W))
    Loss = Lw + Ld
                
    return Loss

#def gradCE(W, b, x, y, reg):
    # Your implementation here

def accuracy(weights, b, data, label):
    trained = np.matmul(data, weights) + b
    correct = 0
    for i in range(len(trained)):
        if (abs(label[i] - trained[i]) < 0.5):
            correct = correct + 1
    return correct/len(trained)
    

def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS, validData, validTarget, testData, testTarget):
    x = trainingData
    y = trainingLabels

    #all the loses of linear gradient descent
    MSEs = np.zeros((iterations, 1))
    MSEvalid = np.zeros((iterations, 1))
    MSEtest = np.zeros((iterations, 1))

    #array of iterations
    iterationNum = np.zeros((iterations, 1))

    #accuracy classifications
    accuracies = np.zeros((iterations, 1))
    print(np.shape(b))
    accuracyInitial = accuracy(W, b, validData, validTarget)

    print(np.shape(b))
    for i in range(iterations):
 #       print(i)
        gradW, gradB = gradMSE(W, b, x, y, reg)
        print(np.shape(gradB))
        W = W - alpha*gradW
        b = b - alpha*gradB
    
        
        MSEs[i] = MSE(W, 0, trainingData, trainingLabels, 0)
        MSEvalid[i] = MSE(W, 0, validData, validTarget, 0)
        MSEtest[i] = MSE(W, 0, testData, testTarget, 0)     
#        accuracies[i] = accuracy(W, b, x, y)

        iterationNum[i] = i
        if (np.all(abs(gradW)) < EPS):
            return W, b
    print(np.shape(b))
    accuracyFinal = accuracy(W, 0, validData, validTarget)
    print(accuracyInitial)
    print(accuracyFinal)

#    plt.xlabel('Iterations')
#    plt.ylabel('Loss')
#    plt.title('0.5 Reg Training Loss')
#    plt.plot(iterationNum, MSEs, 'r')
#    plt.show()
    return W, b

#def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    # Your implementation here

trainData, validData, testData, trainTarget, validTarget, testTarget = loadData();
W = np.zeros((784, 1))
x = trainData
N = len(x)
b=0

x = np.reshape(x, (N, np.shape(x)[1]*np.shape(x)[2]))
testData = np.reshape(testData, (len(testData), np.shape(testData)[1]*np.shape(testData)[2]))
validData = np.reshape(validData, (len(validData), np.shape(validData)[1]*np.shape(validData)[2]))

y = trainTarget

W, b = grad_descent(W, 0, x, y, 0.005, 5000, 0.001, 0.0000001, validData, validTarget, testData, testTarget)




