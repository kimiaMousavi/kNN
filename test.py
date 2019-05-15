import pandas as pd
import numpy as np
import operator
from collections import Counter
import matplotlib.pyplot as plt
import csv

import scipy
from scipy.spatial.distance import cdist

"""
Input file names:

mnist_train.csv
mnist_test.csv

"""


class KNN(object):
    def __init__(self):
        # self.train_data = with open(, newline='') as csvfile:
        self.train_data = np.array([])
        self.test_data =np.array([])
        self.train_lable=[]
        self.test_lable=[]


    def load_train_data(self):
        temp=[]
        # train_lable=[]
        with open('mnist_train.csv','r') as csvFile:
            reader = csv.reader(csvFile)
            reader.next()
            for row in reader:
                self.train_lable.append(row.pop(0))
                temp.append(row)
        self.train_data=np.array(temp)


    def load_test_data(self):
        temp=[]
        # train_lable=[]
        with open('mnist_test.csv','r') as csvFile:
            reader = csv.reader(csvFile)
            reader.next()
            for row in reader:
                self.test_lable.append(row.pop(0))
                temp.append(row)
        self.test_data=np.array(temp)

    def get_distance_matrix_of_test_to_train(self, k=9):
        temp=[]
        i=0
        for test in self.test_data:
            for train in self.train_data:
                print(np.sqrt(np.sum((train - test)**2)))
                temp.append([self.train_lable[i],np.sqrt(np.sum((train - test)**2))])
                i+=1


        # print self.test_data

    def predict_test_data(self, k=9):
        pass

    def plot_learning_curve(self, plot_curve1_dict):
        # Plot your results and save it as a picture.
        pass

    def run(self):
        # Run your algorithm for different Ks.
        self.load_train_data()
        self.load_test_data()
        # print(self.train_lable)
        # print(self.train_data)
        self.get_distance_matrix_of_test_to_train(k=2)


if __name__ == "__main__":
    obj = KNN()
    obj.run()
    # obj.
