#ECE421 Assignment 1
#Danil Ojha & Elizabeth Bertozzi

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
    yhat = 1/(1 + np.exp(-np.matmul(x, W) + b))
    Ld = (1/N)*np.sum((np.dot(-1*y.T, np.log(yhat)) - (np.dot((1-y).T, np.log(1 - yhat)))))
    #Weight decay loss function
    Lw = (reg/2)*np.square(np.linalg.norm(W))
    Loss = Lw + Ld
    return Loss

def gradCE(W, b, x, y, reg):
    N = len(x)
    yhat = 1/(1 + np.exp(-1*np.matmul(x, W) + b))
    gradW = np.zeros((N,1))
    gradW = np.dot(x.T, yhat-y)*(1/N) + reg*W
    gradB = np.mean(yhat-y)
    return gradW, gradB

#finding the accuracy of weights and biases #ofcorrectclassifications/#ofimages
def accuracy(weights, b, data, label, lossType="None"):
    trained = np.matmul(data, weights) + b
    if (lossType == 'CE'):
        trained = 1/(1+np.exp(-1*np.matmul(data, weights) + b))    
    correct = 0
    for i in range(len(trained)):
        if (abs(label[i] - trained[i]) < 0.5):
            correct = correct + 1
    return correct/len(trained)

#"normal equation" of the least squares formula for part 1.5
def normal(x, y):
    a = np.matmul(x.T, x)
    b = np.linalg.inv(a)
    c = np.matmul(b, x.T)
    result = np.matmul(c, y)
    return result

def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS, lossType="None"):
    x = trainingData
    y = trainingLabels
    for i in range(iterations):
        #calculating the gradient and updating the weights
        if (lossType == 'MSE'):
            gradW, gradB = gradMSE(W, b, x, y, reg)
        else:
            gradW, gradB = gradCE(W, b, x, y, reg)
        W = W - alpha*gradW
        b = b - alpha*gradB
        #exit condition if difference is less than provided error
        if (np.all(abs(gradW)) < EPS):
            return W, b
    return W, b

def buildGraph(beta1=None, beta2=None, epsilon=None, lossType="None", learning_rate=None):
    tf.set_random_seed(421)
    #Initialize tensors
    W = tf.Variable(tf.truncated_normal(shape=[784, 1], stddev=0.5), name='weight') #weights
    b = tf.Variable(0, name = 'bias') #bias
    x = tf.placeholder(tf.float32, shape=(None,784), name='x') #data
    y = tf.placeholder(tf.float32, shape=(None, 1), name='y') #real labels 
    reg = tf.placeholder(tf.float32, shape=(1), name ='reg') #regularization
    
    yhat = tf.placeholder(tf.float32, shape=(3500, 1), name = 'yhat') #predicted labels
    yhat = tf.add(tf.cast(tf.matmul(x, W), tf.float32), tf.cast(b, tf.float32))
    
    if lossType == "MSE":
        Loss = tf.losses.mean_squared_error(y,yhat)
    elif lossType == "CE":
        Loss = tf.losses.sigmoid_cross_entropy(y, yhat)

    opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(Loss)
    #Be sure to run opt_op.run() in training
    return W, b, x, yhat, y, Loss, opt, reg

def SGD(Weight, bias, trainingData, trainingTarget, alpha, epochs, regularization, EPS, batchSize, validData, validTarget, testData, testTarget, lossType="None"):
    #Initialize graph
    W, b, x, yhat, y, Loss, opt, reg = buildGraph(beta1=None, beta2=None, epsilon=None, lossType="MSE", learning_rate=0.001)
    init = tf.global_variables_initializer()
    #Calculate number of batches in training set
    N = len(trainingData)
    num_mini_batches = np.floor(N/batchSize)

    trainingLoss = []
    validationLoss = []
    testLoss = []
    trainingAccuracy = []
    validationAccuracy = []
    testAccuracy = []
    iterations =[]
    
    
    with tf.Session() as sess:
        sess.run(init)
        for i in range(epochs):
            #Shuffle dataset each epoch
            permutation = np.random.permutation(N)
            shuffled_x = trainingData[permutation,:]
            shuffled_y = trainingTarget[permutation,:]
            epoch_loss = 0
            print(i)
            iterations.append(i)
            
            for j in range(int(num_mini_batches)):
                #get mini-batches
                mini_batch_x = shuffled_x[j*batchSize : (j+1)*batchSize :,]
                mini_batch_y = shuffled_y[j*batchSize : (j+1)*batchSize :,]
                sess.run([opt, Loss], {x: mini_batch_x, y: mini_batch_y})
                L = MSE(W.eval(), b.eval(), trainingData, trainingTarget, regularization)
                trainingLoss.append(L)
 #               validationLoss.append(sess.run(Loss, feed_dict={validData, validTarget}))
 #               testLoss.append(sess.run(Loss, feed_dict={testData, testTarget}))
              
                
    return iterations, trainingLoss, valdiationLoss, testLoss, trainingAccuracy, validationAccuracy, testAccuracy

  
#loading data
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData();






