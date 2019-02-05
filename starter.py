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
    gradB = (1/N)*(Wx + b - y)
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

def gradCE(W, b, x, y, reg):
    N = len(x)
    gradW = np.zeros((N,1))
    temp = (- y + 1 - (1/(1+np.exp(W.T*x + b))))
    print(temp.shape);
    print(x.shape)
    gradW = (1/N)*(np.matmul(x.T, temp)) + reg*W
    gradB = (1/N)*(1 - y - (1/(1+np.exp(W.T*x + b))))

    return gradW, gradB

def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS):
    x = trainingData
    y = trainingLabels
    for i in range(iterations):
        #print(i)
        gradW, gradB = gradMSE(W, b, x, y, reg)
        Wold = W
        W = W - alpha*gradW
        b = b - alpha*gradB
        if (np.any(abs(W - Wold)) < EPS):
            return W, b
    return W, b

def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):

    #Initialize tensors
    W = tf.truncated_normal(shape=[784], stddev=0.5, dtype=tf.float32) #weights
    b = tf.Variable() #bias
    x = tf.placeholder(tf.float32, shape=(3500,784)) #data
    yhat = tf.placeholder(tf.float32, shape=(3500, 1)) #predicted labels
    y = tf.placeholder(tf.float32, shape=(3500, 1)) #real labels
    reg = tf.placeholder(tf.float32, shape=(1)) #regularization param
    
    tf.set_random_seed(421)

    if loss == "MSE":
    # Your implementation
        L = MSE(W, b, x, y, reg)
    elif loss == "CE":
        L = crossEntropyLoss(W, b, x, y, reg)

    opt = tf.train.AdamOptimizer(learning_rate=0.001,beta1, beta2, epsilon)
    opt_op = opt.minimize(L)
    #Be sure to run opt_op.run() in training
    return W, b, yhat, y, L, opt, reg

def SDG(W, b, x, yhat, y, L, opt, reg, batchSize, epoch)
    N = len(x)
    num_mini_batches = math.floor(N/batchSize)
    _permutation = np.random.permutation(N)
    for i in range(epochs)
        shuffled_x = x[:,_permutation]
        shuffled_y =
        
        for j in range(num_mini_batches)
            #get mini-batches
            mini_batch_x = shuffled_x[:,j*batchSize : (j+1)*batchSize]
            mini_batch_y = shuffled_y[:,j*batchSize : (j+1)*batchSize]
            #Calculate with update with mini-batch
            Loss = 





        #Store the training, validation and test losses and accuracies



trainData, validData, testData, trainTarget, validTarget, testTarget = loadData();
W = np.zeros((784, 1))
x = trainData
N = len(x)
x = np.reshape(x, (N, np.shape(x)[1]*np.shape(x)[2]))

testData = np.reshape(testData, (len(testData), np.shape(testData)[1]*np.shape(testData)[2]))
validData = np.reshape(validData, (len(validData), np.shape(validData)[1]*np.shape(validData)[2]))

y = trainTarget

trainingOld = MSE(W, 0, x, y, 0)
testOld = MSE(W, 0, testData, testTarget, 0)
validOld = MSE(W, 0, validData, validTarget, 0)

W, b = grad_descent(W, 0, x, y, 0.0001, 5000, 0, 0.0000001)

LossTraining = MSE(W, 0, x, y, 0)
print(1 - trainingOld)
print(1 - LossTraining)

LossTest = MSE(W, 0, testData, testTarget, 0)
print(1- testOld)
print(1 - LossTest)

LossValid = MSE(W, 0, validData, validTarget, 0)
print(1 - validOld)
print(1- LossValid)






