
import pandas as pd
import numpy as np
from collections import Counter

class NaiveBayesClassifierBernoulli(object):
    def __init__(self):
        self.__data__ = None
        self.__split_data__ = None
        self.__priors__ = None
        self.__class_probability__ = None
        self.__feature_probability__ = None

    def __get_model__(self):
        return self.__feature_probability__

    def printData(self):
        print(self.__data__.head())

    def openData(self, io, sheet_name):
        self.__data__ = pd.read_excel(io, sheet_name=sheet_name)

    def split_class(self, split_criteria):
        self.__data__['labels'] = self.__data__['labels'].map(split_criteria)
        labels = set(self.__data__['labels'])
        self.__split_data__ = {k:[[row.gradient, row['1st-peak'], row['last-peak']] for index, row in self.__data__.iterrows() if row.labels == k]  for k in labels}

    def class_probability(self):
        N = 0
        for key in self.__split_data__.keys():
            N += len(self.__split_data__[key])

        print(f"length N: {N}")
        self.__class_probability__ = {k:0 for k in self.__split_data__.keys()}
        for el in self.__class_probability__.keys():
            self.__class_probability__[el] = len(self.__split_data__[el])/N
            print(f"Probability for Class {el}: {self.__class_probability__[el]}")

    def feature_probability(self):
        self.__feature_probability__ = {k:{} for k in self.__split_data__.keys()}
        for _class in self.__split_data__.keys():
            feature_length = len(self.__split_data__[_class][0])
            feature_matrix = np.array(self.__split_data__[_class])
            for i in range(feature_length):
                summation = np.sum(feature_matrix[:, i])
                self.__feature_probability__[_class][f"X{i}"] = summation/len(self.__split_data__[_class])
                print(f"Probability for Class {_class} and feature X{i}: {self.__feature_probability__[_class][f'X{i}']}")

    def predict(self, row, feature_map = {}):
        '''predicts form a passed in row'''
        features = self.__feature_probability__
        classes  = self.__class_probability__


        _max = 0
        _feature_key = None
        '''go through each class and determine the prob'''
        for _class in self.__class_probability__.keys():
            _prob = classes[_class]
            _row_index = 0

            #this assumes that all features are present
            for _feature in features[_class].keys():
                _prob *= features[_class][_feature] if row[0] == 1 else 1. -features[_class][_feature]

            if(_prob >= _max):
                _max = _prob
                _feature_key = _class

        print(f"Predicted Class: {feature_map[_feature_key]}")
        print(f"Probability: {_max}")

class NaiveBayesClassifierGaussian(NaiveBayesClassifierBernoulli):
    def __init__(self):
        pass

    

if __name__ == "__main__":
    nb = NaiveBayesClassifierBernoulli()

    filename = '../decision_tree/decision_data.xlsx'
    nb.openData(filename, 'no lens')
    nb.split_class({'right':1, 'left':0})
    nb.class_probability()
    nb.feature_probability()

    nb.predict([0,0,1], {1:'right', 0:'left'})

