import numpy as np 
from helper import *
import math
'''
Homework2: logistic regression classifier
'''


def logistic_regression(data, label, max_iter, learning_rate):
    '''
    The logistic regression classifier function.
    Args:
        data: train data with shape (1561, 3), which means 1561 samples and 
        each sample has 3 features.(1, symmetry, average internsity)
    label: train data's label with shape (1561,1). 
        1 for digit number 1 and -1 for digit number 5.
        max_iter: max iteration numbers
        learning_rate: learning rate for weight update	
        Returns:
            w: the seperater with shape (3, 1). You must initilize it with w = np.zeros((d,1))
            '''
    n, d=data.shape
    w=np.zeros((d,1))
    for i in range(max_iter):
        w[:,0]-=learning_rate*(gradient(data, label,w))
    return w

def gradient(data,label, w):
    n,d=data.shape
    E=0
    for i in range(n):
        E+=((-1/n)*label[i]*data[i,:])/(1+math.exp(label[i]*np.matmul(np.transpose(w),data[i])))
    return E

def thirdorder(data):
    '''
    This function is used for a 3rd order polynomial transform of the data.
    Args:
        data: input data with shape (:, 3) the first dimension represents 
        total samples (training: 1561; testing: 424) and the 
        second dimesion represents total features.

    Return:
        result: A numpy array format new data with shape (:,10), which using 
        a 3rd order polynomial transformation to extend the feature numbers 
        from 3 to 10. 
        The first dimension represents total samples (training: 1561; testing: 424) 
        and the second dimesion represents total features.
        '''
    N, _ = data.shape
    newData = np.ones((N, 1))
    newData = np.append(newData, data, axis=1)
    for i in range(2, 4):
        for j in range(0, i + 1):
            col = np.reshape(
                np.array(data[:, 0]**(i - j) * data[:, 1]**j), (N, 1))
            newData = np.append(newData, col, axis=1)
    return newData


def accuracy(x, y, w):
    '''
    This function is used to compute accuracy of a logsitic regression model.
    
    Args:
    x: input data with shape (n, d), where n represents total data samples and d represents
        total feature numbers of a certain data sample.
    y: corresponding label of x with shape(n, 1), where n represents total data samples.
    w: the seperator learnt from logistic regression function with shape (d, 1),
        where d represents total feature numbers of a certain data sample.

    Return 
        accuracy: total percents of correctly classified samples. Set the threshold as 0.5,
        which means, if the predicted probability > 0.5, classify as 1; Otherwise, classify as -1.
    '''
    n,d=x.shape
    miss=0
    temp=0
    for i in range(n):
        if (1.0/(1+np.exp(-(np.dot(np.transpose(w), np.transpose(x[i,:])))))>0.5):
            temp=1
        else:
            temp=-1
        if temp!=y[i]:
            miss+=1
    return (n-miss)/n
            


