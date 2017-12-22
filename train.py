#!/Users/wuyou/anaconda2/envs/py3/
# -*- coding: utf-8 -*-

import skimage
from skimage import io, transform
import numpy as np
import math
import os

import pickle

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report

if __name__ == "__main__":
    # write your code here
    original_face_dir = "datasets/original/face"
    original_nonface_dir = "datasets/original/nonface"
    resized_grey_face_dir = "datasets/resized_grey/face"
    resized_grey_nonface_dir = "datasets/resized_grey/nonface"
    
    #读入图像数据并灰度化和缩放成24*24的大小
    '''
    imgs_face = os.listdir(original_face_dir)
    imgs_nonface = os.listdir(original_nonface_dir)
    
    img_num = len(imgs_face)
    for i in range(img_num): 
        img_grey = io.imread(original_face_dir + '/' + imgs_face[i], as_grey=True)
        img_grey_resized = transform.resize(img_grey,(24,24))
        io.imsave(resized_grey_face_dir + '/face_'+'{0:03}'.format(i)+'.jpg',img_grey_resized)
    
    for i in range(img_num): 
        img_grey = io.imread(original_nonface_dir  + '/' + imgs_nonface[i], as_grey=True)
        img_grey_resized = transform.resize(img_grey,(24,24))
        io.imsave(resized_grey_nonface_dir+ '/nonface_'+'{0:03}'.format(i)+'.jpg',img_grey_resized)
    '''
    
    #提取feature并制作对应的文件
    '''
    imgs_facegrey = os.listdir(resized_grey_face_dir)
    imgs_nonfacegrey = os.listdir(resized_grey_nonface_dir)
    print (imgs_facegrey)
    print (imgs_nonfacegrey)
    
    #by testing we can know npd extract 165600 dimension features
    feature_sets = np.zeros(shape=(len(imgs_facegrey)+len(imgs_nonfacegrey), 165600))
    
    for i in range(len(imgs_facegrey)):
        img = io.imread(resized_grey_face_dir + '/' + imgs_facegrey[i])
        print (i)
        print (img.shape)
        npd = NPDFeature(img)
        feature = npd.extract()
        feature_sets[i] = feature
        #print (i)
        
    for i in range(len(imgs_nonfacegrey)):
        img = io.imread(resized_grey_nonface_dir + '/' + imgs_nonfacegrey[i])
        print (i)
        print (img.shape)
        npd = NPDFeature(img)
        feature = npd.extract()
        feature_sets[len(imgs_facegrey)+i] = feature
        #print (len(imgs_facegrey)+i)
    
    #dump feature文件
    output = open('datasets/features.pk', 'wb')
    pickle.dump(feature_sets, output)
    output.close()
    print("features.pk created!")
    '''
    
    #读入feature文件并制作数据集，设置训练数据的权重参数向量
    imgs_facegrey = os.listdir(resized_grey_face_dir)
    imgs_nonfacegrey = os.listdir(resized_grey_nonface_dir)
    pk_file = open('datasets/features.pk', 'rb')
    X = pickle.load(pk_file)
    y = [1] * len(imgs_facegrey) + [-1] * len(imgs_nonfacegrey)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
    #转换list为numpy数组
    train_y = np.array(train_y)
    test_y = np.array(test_y)
    
    '''It is test demo, just ignore.
    #训练样本的权重参数
    weight = np.ones(train_X.shape[0]) / float(train_X.shape[0])
    #若分类器集合以及个数
    weak_classifier_sets = []
    n_weakers_limit = 5;
    #样本权重调节参数
    alpha = []
    for i in range(n_weakers_limit):
        clf = DecisionTreeClassifier(max_depth=2)
        #根据带权重的样本训练决策树
        weak_classifier_sets.append(clf.fit(train_X,train_y,sample_weight=weight))
        train_y_calcu = weak_classifier_sets[i].predict(train_X)
        # train_y_calcu*train_y 同号得到1，异号得到-1
        error = 1.0 - np.sum(np.maximum(0, train_y_calcu*train_y)) / float(train_y.shape[0])
        print ("DST_" + str(i))
        print (error)
        alpha.append(0.5 * math.log( (1-error) / error ))
        weight = weight * np.exp(-train_y_calcu*train_y * alpha[i])
        weight = weight * 1.0/np.sum(weight)
    
    alpha = np.array(alpha)
    proba_sets = []
    for i in range(n_weakers_limit):
        proba_sets.append(weak_classifier_sets[i].predict(test_X))
    proba_sets = np.array(proba_sets)
    
    ab = []
    for i in range(n_weakers_limit):
        ab.append(alpha[i] * proba_sets[i])
    ab = np.array(ab)
    print (np.sum(ab, axis=0))
    test_y_calcu = np.sign(np.sum(ab, axis=0))
    print (test_y_calcu) 
    error = 1.0 - np.sum(np.maximum(0, test_y_calcu*test_y)) / float(test_y.shape[0])
    print ("Adaboost")
    print (error)
    '''
    
    #Adaboost
    classifier = AdaBoostClassifier(DecisionTreeClassifier, 5)
    classifier.fit(train_X, train_y)
    pre_scr = classifier.predict_scores(test_X)
    pre = classifier.predict(test_X)
    error = 1.0 - np.sum(np.maximum(0, pre*test_y)) / float(test_y.shape[0])
    #print (error)
    
    target_names = ['nonface', 'face']
    #print(classification_report(test_y, pre, target_names=target_names))
    
    report = open('report.txt', 'wb')
    report.write(classification_report(test_y, pre, target_names=target_names))
    report.close( )
    #print("report.txt created!")
