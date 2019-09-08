import math
import numpy as np
import csv
import random

def load_dataset(path):
    dataset = []
    with open(path) as file:
        reader = csv.reader(file)
        for row in reader:
            dataset.append(row)
    return dataset

def str_to_float(dataset):
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            dataset[i][j] = float(dataset[i][j])

def minMaxFunction():
    pass

def normalization():
    pass

def crossValidation():
    pass

def accuracy_metrics():
    pass

def prediction():
    pass

def loss():
    pass

def gradientDescent():
    pass

def logisticRegression():
    pass

def evaluateAlgorithm():
    pass

filename = 'data.csv'
dataset = load_dataset(filename)
str_to_float(dataset)