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


def gradCE(W, b, x, y, reg):
    N = len(x)
    gradW = np.zeros((N,1))
    temp = (- y + 1 - (1/(1+np.exp(W.T*x + b))))
    print(temp.shape);
    print(x.shape)
    gradW = (1/N)*(np.matmul(x.T, temp)) + reg*W
    gradB = (1/N)*(1 - y - (1/(1+np.exp(W.T*x + b))))

    return gradW, gradB

#finding the accuracy of weights and biases #ofcorrectclassifications/#ofimages
def accuracy(weights, b, data, label):
    trained = np.matmul(data, weights) + b
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

    #initial accuracies
    accuracyInitial = accuracy(W, b, trainingData, trainingLabels)
    accuracyInitialV = accuracy(W, b, validData, validTarget)
    accuracyInitialT = accuracy(W, b, testData, testTarget)

    for i in range(iterations):
        #calculating the gradient and updating the weights
        gradW, gradB = gradMSE(W, b, x, y, reg)
        W = W - alpha*gradW
        b = b - alpha*gradB

        #Losses and accuracies stored for each iteration
        #comment out what is not needed for graph
        iterationNum[i] = i
        MSEs[i] = MSE(W, b, trainingData, trainingLabels, 0)
        MSEvalid[i] = MSE(W, b, validData, validTarget, 0)
        MSEtest[i] = MSE(W, b, testData, testTarget, 0)     
        accuracies[i] = accuracy(W, b, x, y)

        #exit condition if difference is less than provided error
        if (np.all(abs(gradW)) < EPS):
            return W, b

    #final accuracies
    accuracyFinal = accuracy(W, b, trainingData, trainingLabels)
    accuracyFinalV = accuracy(W, b, validData, validTarget)
    accuracyFinalT = accuracy(W, b, testData, testTarget)

    #printing out initial and final accuracies for all data sets
    #after running descent on training data
    print("Training accuracy Test 0.5 Reg")
    print(accuracyInitial)
    print(accuracyFinal)
    print("Valid")
    print(accuracyInitial)
    print(accuracyFinal)
    print("Test")
    print(accuracyInitial)
    print(accuracyFinal)

    #plotting accuracies
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('0.5 Reg Training Accuracy')
    plt.plot(iterationNum, accuracies, 'r')
    plt.show()
    
    return W, b, MSEs, MSEvalid, MSEtest, iterationNum

def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):

    #Initialize tensors
    W = tf.truncated_normal(shape=[784], stddev=0.5, dtype=tf.float32) #weights
    b = tf.Variable(1) #bias
    x = tf.placeholder(tf.float32, shape=(3500,784)) #data
    yhat = tf.placeholder(tf.float32, shape=(3500, 1)) #predicted labels
    y = tf.placeholder(tf.float32, shape=(3500, 1)) #real labels
    reg = tf.placeholder(tf.float32, shape=(1)) #regularization param
    
    tf.set_random_seed(421)

    if loss == "MSE":
        L = MSE(W, b, x, y, reg)
    elif loss == "CE":
        L = crossEntropyLoss(W, b, x, y, reg)

    opt = tf.train.AdamOptimizer(learning_rate=0.001, beta1, beta2, epsilon)
    opt_op = opt.minimize(L)

    #Initialize variables? I think
    sess.run(tf.global_variables_initializer())
    #Be sure to run opt_op.run() in training
    return W, b, yhat, y, L, opt, reg

def SGD(W, b, x, yhat, y, L, opt, reg, batchSize, epoch)
    #Initialize graph
    buildGraph(beta1=None, beta2=None, epsilon=None, lossType="MSE", learning_rate=0.001)
    #Calculate number of batches in training set
    N = len(x)
    num_mini_batches = math.floor(N/batchSize)
    _permutation = np.random.permutation(N)
    for i in range(epochs)
        #Shuffle dataset each epoch
        shuffled_x = x[:,_permutation]
        shuffled_y = y[:,_permutation]

        
        for j in range(num_mini_batches):
            #get mini-batches
            mini_batch_x = shuffled_x[:,j*batchSize : (j+1)*batchSize]
            mini_batch_y = shuffled_y[:,j*batchSize : (j+1)*batchSize]
            #Calcuate step
            sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
            #Update with mini-batch
            
            #Loss = tf.losses.mean_squared_error(labels=mini_batch_y,predictions=yhat,weights=W, loss_collection=tf.GraphKeys.LOSSES, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS)
            sess.run()






        #Store the training, validation and test losses and accuracies


#loading data
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData();

#initializing parameters
W = np.zeros((784, 1))
x = trainData
N = len(x)
b = 0
epochs = 5000
error = 0.0000001


#reshaping data
x = np.reshape(x, (N, np.shape(x)[1]*np.shape(x)[2]))
testData = np.reshape(testData, (len(testData), np.shape(testData)[1]*np.shape(testData)[2]))
validData = np.reshape(validData, (len(validData), np.shape(validData)[1]*np.shape(validData)[2]))

y = trainTarget

#more parameter initializiation
LR = 0.005
reg = 0

####normal function (part 1.5)###
n = normal(x, y)
initialMSE = MSE(W, b, x, y, reg)
finalMSE = MSE(n, b, x, y, reg)
print("Training MSE")
print(initialMSE)
print(finalMSE)
accuracyinitial = accuracy(W, b, x, y)
accuracyfinal = accuracy(n, b, x, y)
print("Training Accuracy")
print(accuracyinitial)
print(accuracyfinal)
#################################

###running gradient descent and plotting results#####
W, b, Loss, LossValid, LossTest, iterations = grad_descent(W, b, x, y, LR, epochs, reg, error, validData, validTarget, testData, testTarget)
print("Training")
print(Loss[0])
print(Loss[4999])
print("Valid")
print(LossValid[0])
print(LossValid[4999])
print("Test")
print(LossTest[0])
print(LossTest[4999])

plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('0.5 Reg Training Loss')
plt.plot(iterations, Loss, 'r')
plt.show()

plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('0.5 Reg Valid Loss')
plt.plot(iterations, LossValid, 'r')
plt.show()

plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('0.5 Reg Test Loss')
plt.plot(iterations, LossTest, 'r')
plt.show()
###############################################



