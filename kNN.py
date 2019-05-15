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
        self.train_data = np.genfromtxt('mnist_train.csv', delimiter=',')
        self.test_data = np.genfromtxt('mnist_test.csv', delimiter=',')


    def load_train_data(self):
        return self.train_data


    def load_test_data(self):

        return self.test_data


    def get_distance_matrix_of_test_to_train(self, k=9):
        temp=[]
        for i in range(1,len(self.test_data)):
            for j in range(1,len(self.train_data)):
                labale=self.train_data[j][0]
                self.train_data=np.delete(self.train_data, np.s_[::2], 1)
                temp.append(np.sqrt(np.sum((self.train_data[j] - self.test_data[i])**2)))


        # print self.test_data

    def predict_test_data(self, k=9):
		pass

    def plot_learning_curve(self, plot_curve1_dict):
		# Plot your results and save it as a picture.
		pass

    def run(self):
		# Run your algorithm for different Ks.
		self.get_distance_matrix_of_test_to_train(k=2)


if __name__ == "__main__":
    obj = KNN()
    obj.run()
    # obj.load_train_data()
