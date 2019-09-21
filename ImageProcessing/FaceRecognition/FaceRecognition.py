import numpy as np
from sklearn.model_selection import train_test_split
import pickle

user = np.load('user_1.npy')
unknown = np.load('unknown.npy')

user = user.reshape(user.shape[0],-1)
unknown = unknown.reshape(unknown.shape[0],-1)

dataset = np.concatenate((user,unknown))
dataset = dataset/255.

labels = np.zeros((dataset.shape[0],1))
labels[400:,:] = 1.0

def accuracy_metrics(actual, pred):
    count = 0
    for i in range(len(actual)):
        if actual[i] == pred[i]:
            count += 1
    return count / len(actual) * 100

def prediction(x, w):
    z = np.dot(x, w)
    return 1 / (1 + np.exp(-z))

def cost_function(features, target, weights):
    n = len(target)
    y_pred = prediction(features, weights)
    cost_1 = -target * np.log(y_pred)
    cost_2 = -(1 - target) * np.log((1 - y_pred))
    cost = cost_1 + cost_2
    total = np.sum(cost) / n
    return total

def gradientDescent(x, y, epochs, alpha):
    w = np.zeros(x.shape[1])
    n = len(x)
    for epoch in range(epochs):
        pred = prediction(x, w)
        y_pred = list(pred)
        loss = pred - y
        for i in range(len(y_pred)):
            y_pred[i] = round(y_pred[i])
        score = accuracy_metrics(y, y_pred)
        print("Accuracy score ::",score)
        grad = (2 / n) * (x.T.dot(loss))
        w = w - alpha * grad
        if epoch % 1000 == 0:
            cost = cost_function(x, y, w)
            print("Cost is", cost)
    return w

def logisticRegression(x_train, y_train, x_test, y_test, epochs, alpha):
    weights = gradientDescent(x_train, y_train.flatten(), epochs, alpha)
    file = open('weights.pkl','wb')
    pickle.dump(weights,file)
    file.close()
    y_pred = prediction(x_test, weights)
    for i in range(len(y_pred)):
        y_pred[i] = round(y_pred[i])
    score = accuracy_metrics(y_test, y_pred)
    return score

def evaluateAlgorithm(dataset,labels, epochs, alpha):
    x_train,x_test,y_train,y_test = train_test_split(dataset,labels,test_size=0.25)
    score = logisticRegression(x_train,y_train,x_test,y_test,epochs,alpha)
    return score

epochs = 10000
alpha = 0.0001
acc = evaluateAlgorithm(dataset,labels, epochs, alpha)
print(acc)
