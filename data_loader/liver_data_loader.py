# -*- coding:utf-8 -*-
from base.base_data_loader import BaseDataLoader

import os
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import tensorflow as tf 
from keras.utils import np_utils 
from SampleData import sampledata

class LiverDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(LiverDataLoader, self).__init__(config)
        self.dict = {'A': 0, 'B': 1, 'K': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'J': 8, \
                     'E5':range(9,14), 'F5':range(14,19), 'G5':range(19,24), 'H5':range(24,29), 'I5':range(29,34), 'J5':range(34,39)}
        self.path = config.data_path
        self.Num_train = config.Num_train
        self.Num_test = config.Num_test
        (self.X_train, self.y_train), (self.X_test, self.y_test) = self.loadSampledData() 

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test
    
    def get_data(self, Fusion):
        dict = self.dict
        Train_list = []
        Test_list = []
        (trainData, trainLabels), (testData, testLabels) = (self.X_train, self.y_train), (self.X_test, self.y_test)
            
        ############
        # Label Smooth
        # new_labels = (1.0 - label_smoothing) * one_hot_labels + label_smoothing / classes
                
        # # 不转置            
        # for modual in Fusion:
            # if len(modual) == 1:  
                # input_shape = (trainData.shape[1], trainData.shape[2], 1)
                # Train_list.append(trainData[:,:,:,dict[modual]].reshape(trainData.shape[0], input_shape[0], input_shape[1], 1))
                # Test_list.append(testData[:,:,:,dict[modual]].reshape(testData.shape[0], input_shape[0], input_shape[1], 1)) 
            # else:  
                # input_shape = (trainData.shape[1], trainData.shape[2], 5, 1) 
                # Train_list.append(trainData[:,:,:,dict[modual]].reshape(trainData.shape[0], input_shape[0], input_shape[1], 5, 1))
                # Test_list.append(testData[:,:,:,dict[modual]].reshape(testData.shape[0], input_shape[0], input_shape[1], 5, 1))
                
                
        # 转置数据*********************************
        for modual in Fusion:
            if len(modual) == 1:  
                input_shape = (trainData.shape[1], trainData.shape[2], 1)
                Train_list.append(trainData[:,:,:,dict[modual]].reshape(trainData.shape[0], input_shape[0], input_shape[1]))
                Test_list.append(testData[:,:,:,dict[modual]].reshape(testData.shape[0], input_shape[0], input_shape[1])) 
            else:  
                input_shape = (trainData.shape[1], trainData.shape[2], 5, 1)
                Train_list.append(trainData[:,:,:,dict[modual]].reshape(trainData.shape[0], input_shape[0], input_shape[1], 5))
                Test_list.append(testData[:,:,:,dict[modual]].reshape(testData.shape[0], input_shape[0], input_shape[1], 5))
                
        if len(modual) == 1 or len(modual) == 3:         
            Train_list = tf.cast(tf.stack(Train_list, axis=3), tf.float32)
            Test_list = tf.cast(tf.stack(Test_list, axis=3), tf.float32)
        else:
            Train_list = tf.cast(tf.stack(Train_list, axis=4), tf.float32)
            Test_list = tf.cast(tf.stack(Test_list, axis=4), tf.float32)
            # # 转置
            Train_list = tf.expand_dims(tf.transpose(Train_list, perm=[0,1,2,4,3]),5)
            Test_list = tf.expand_dims(tf.transpose(Test_list, perm=[0,1,2,4,3]),5)
                
            Train_list = tf.unstack(Train_list, axis=4)
            Test_list = tf.unstack(Test_list, axis=4)
            
        with tf.Session() as sess:
            for i, arr in enumerate(Train_list):
                Train_list[i] = arr.eval()
            for i, arr in enumerate(Test_list):
                Test_list[i] = arr.eval()               
                
        # print '*'*5, len(Train_list), Test_list[0].shape
        # raw_input()
        #*********************************************** 
        return Train_list, trainLabels, Test_list, testLabels
  
    def loadSampledData(self):
        path = str(self.path)
        Num_train = self.Num_train
        Num_test = self.Num_test
        sampledata(path+'/train.txt', Num_train, 0, path+'/trainSample.txt') 
        sampledata(path+'/test.txt', Num_test, 0, path+'/testSample.txt')    
        
        train_data = []  
        train_labels = [] 
        max_l = -1    
        fp = open(os.path.join(path, "trainSample.txt"), 'r')
        line = fp.readline()   # 调用文件的 readline() 
        while len(line):
            # print '*'*10,line      
            Level = int(line[0])
            if Level > max_l:
                max_l = Level
            imgpath = line[2:-1]
            mat = sio.loadmat(imgpath)
            # print mat.keys()
            train_data.append(mat['P'])
            train_labels.append(Level)
            line = fp.readline()
        fp.close()
        train_labels = np_utils.to_categorical(train_labels, max_l+1)
        train_data = np.asarray(train_data, dtype="float32")
        print '    Train: ', train_data.shape   

        test_data = []  
        test_labels = []  
        max_l = -1
        fp = open(os.path.join(path, "testSample.txt"), 'r')
        line = fp.readline()   # 调用文件的 readline()  
        while len(line):
            # print '*'*10,line      
            Level = int(line[0])
            if Level > max_l:
                max_l = Level
            imgpath = line[2:-1]
            mat = sio.loadmat(imgpath)
            # print mat.keys()
            test_data.append(mat['P'])
            test_labels.append(Level)
            line = fp.readline()
        fp.close()
        test_labels = np_utils.to_categorical(test_labels, max_l+1)
        test_data = np.asarray(test_data, dtype="float32")
        print '    Test: ', test_data.shape
        
        (X_train, y_train), (X_test, y_test) = (train_data, train_labels), (test_data, test_labels) 
        return (X_train, y_train), (X_test, y_test)