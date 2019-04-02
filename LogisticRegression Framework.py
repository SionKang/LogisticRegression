#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 20:32:35 2019

@author: sionkang
"""

import numpy as np
import pandas as pd
from scipy.optimize import fmin_tnc
from sklearn.model_selection import train_test_split

#   splits the given data into train and test data, depedning on the size.
def tt_split(dataset, split):
    train, test = train_test_split(data, test_size=0.2)
    
    train_X = train.iloc[:, :-1]
    test_X = test.iloc[:, :-1]
    train_Y = train.iloc[:, -1]
    test_Y = test.iloc[:, -1]
    
    return train_X, test_X, train_Y, test_Y

#   utilizes the sigmoid function that puts any numerical value within 0 - 1.
def sigmoid(z):
   
   s = 1/(1+np.exp(-z))
   
   return s
#   using dot product of x data and theta, found the weighted sum.
def net_input(theta, x):
    
    weightedSum = np.dot(x, theta)
    
    return weightedSum

#   using the theta, and x data, forms the probability that is created through weighted sum and sigmoid function.
def probability(theta, x):
    
    weightedSum = net_input(theta, x) 
    
    s = sigmoid(weightedSum)
    
    return s

#   using theta, x data, and y data, finds the total cost of model in comparison to real output.
def cost_function(theta, x, y):
    
    m = len(x)
    s = probability(theta, x)
    J = 0.0
    
    J = y*np.log(s) + (1-y)*np.log(1-s)
    J = sum(J)
    J = J*(-1/m)
   
    return J

#   using theta, x data, and y data, finds the gradient descent in which minima occurs.
def gradient(theta, x, y):
    
    m = len(x)
    s = probability(theta, x)
    g = 0.0
    
    g = (1/m)*np.dot(x.T, s-y)
    
    return g

#   find the optimizing parameters using fmin_tnc function.
def fit(x, y, theta):
    
    optimum = 0.0
    
    optimum = fmin_tnc(cost_function, theta, gradient, (x, y.flatten()), disp=0)
    
    return optimum[0]
    
#   takes the train data and optimizing parameters to create probability between 0 and 1.
def predict(x, optimum):
    s = probability(optimum, x)
    
    return s

#   using the predict function, it converts all numbers into a boolean based on the threshold
#   and finds the accuracy rate in comparison to the actual output.
def accuracy(x, actual_classes, optimum, probab_threshold=0.5):
    s = predict(x, optimum)
    
    boolean = (s >= 0.5).astype(int)
    
    boolean = boolean.flatten()

    a = np.mean(boolean == actual_classes)

    return a*100

#   organizes all code required for logistic regression as one simple code
#   produces accuracy rate from train data, then takes the optimizing parameters created by
#   train data to produce accuracy rate of predicted outputs.
def evaluateData (data, split):
    train_X, test_X, train_Y, test_Y = tt_split(data, split)
    
    train_X = np.c_[np.ones((train_X.shape[0], 1)), train_X]
    train_Y = train_Y[:, np.newaxis]
    test_X = np.c_[np.ones((test_X.shape[0], 1)), test_X]
    test_Y = test_Y[:, np.newaxis]
    theta = np.zeros((train_X.shape[1], 1))
    
    training_fit = fit(train_X, train_Y, theta)
    
    training_accuracy = accuracy(train_X, train_Y.flatten(), training_fit)
    
    
    testing_accuracy = accuracy(test_X, test_Y.flatten(), training_fit)
 
    return training_accuracy, testing_accuracy

#TESTCLASS 
data = pd.read_csv("LogisticData.txt",header=None)
split = 0.2

training_accuracy, testing_accuracy = evaluateData(data, split)

print()
print("Train Data:")
print("Accuracy: " + str(round(np.asscalar(training_accuracy),2)) + "%")
print()
print("Test Data:")
print("Accuracy: " + str(round(np.asscalar(testing_accuracy),2)) + "%")
