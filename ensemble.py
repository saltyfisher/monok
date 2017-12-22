#!/Users/wuyou/anaconda2/envs/py3/
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import math
from sklearn.tree import DecisionTreeClassifier

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_classifier = weak_classifier
        self.n_weakers_limit = n_weakers_limit
        #弱分类器集合
        self.weak_classifier_sets = []
        #弱分类器权重参数
        self.alpha = np.zeros(n_weakers_limit)

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        #训练样本的权重参数
        weight = np.ones(X.shape[0]) / float(X.shape[0])   
        for i in range(self.n_weakers_limit):
            clf = DecisionTreeClassifier(max_depth=5)
            #根据带权重的样本训练决策树
            self.weak_classifier_sets.append(clf.fit(X,y,sample_weight=weight))
            y_calcu = self.weak_classifier_sets[i].predict(X)
            # train_y_calcu*train_y 同号得到1，异号得到-1
            error = 1.0 - np.sum(np.maximum(0, y_calcu*y)) / float(y.shape[0])
            #print ("DST_" + str(i))
            #print (error)
            self.alpha[i] = 0.5 * math.log( (1-error) / error )
            weight = weight * np.exp(-y_calcu*y * self.alpha[i])
            weight = weight * 1.0/np.sum(weight)

    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        proba_sets = []
        for i in range(self.n_weakers_limit):
            proba_sets.append(self.weak_classifier_sets[i].predict(X))
        proba_sets = np.array(proba_sets)
        
        ab = []
        for i in range(self.n_weakers_limit):
            ab.append(self.alpha[i] * proba_sets[i])
        ab = np.array(ab)
        pre_scr = np.sum(ab, axis=0)
        return pre_scr
    
    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        proba_sets = []
        for i in range(self.n_weakers_limit):
            proba_sets.append(self.weak_classifier_sets[i].predict(X))
        proba_sets = np.array(proba_sets)
        
        ab = []
        for i in range(self.n_weakers_limit):
            ab.append(self.alpha[i] * proba_sets[i])
        ab = np.array(ab)
        pre = np.sign(np.sum(ab, axis=0))
        return pre

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
