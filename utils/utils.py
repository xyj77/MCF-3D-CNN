# -*- coding:utf-8 -*-
import os
import csv
import itertools
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import interp

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.utils import shuffle

from keras import backend as K  

import tensorflow as tf 
from PIL import Image
from keras.utils import np_utils    
from keras.utils.vis_utils import plot_model 
import argparse


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args

def accuracy_curve(h, dataset, isSample, save_tag):
    acc, loss, val_acc, val_loss = h.history['acc'], h.history['loss'], h.history['val_acc'], h.history['val_loss']
    epoch = len(acc)
    
    # # 绘图
    plt.figure(figsize=(17, 5))
    plt.subplot(211)
    plt.plot(range(epoch), acc, label='Train')
    plt.plot(range(epoch), val_acc, label='Test')
    plt.title('Accuracy over ' + str(epoch) + ' Epochs', size=15)
    plt.legend(loc = 'lower right')
    plt.grid(True)
    plt.subplot(212)
    plt.plot(range(epoch), loss, label='Train')
    plt.plot(range(epoch), val_loss, label='Test')
    plt.title('Loss over ' + str(epoch) + ' Epochs', size=15)
    plt.legend(loc = 'upper right')
    plt.grid(True)
    plt.savefig('experiments/img/'+ save_tag + '_acc.png')
    # plt.show()
    plt.close('all') # 关闭图
    
    # python2可以用file替代open
    # save_file = dataset + '.txt'
    # f = open(save_file,'ab')
    # f.write('sample:'+str(isSample)+'\n')
    # f.write('acc:'+str(acc)+'\n')
    # f.write('val_acc:'+str(val_acc)+'\n')
    
    # f.write('loss:'+str(loss)+'\n')
    # f.write('val_loss:'+str(val_loss)+'\n\n')
    # f.close()    

def plot_confusion_matrix(cm, classes,
                          save_tag = '',
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('experiments/img/'+ save_tag + '_cfm.png')
    plt.close('all') # 关闭图    

def plot_roc_curve(y_true, y_pred, classes, save_tag):
    # # 绘制ROC曲线
    if len(classes) == 2:
        fpr, tpr, thresholds = roc_curve(y_true[:,1], y_pred[:,1])
        fpr[0], tpr[0] = 0, 0
        fpr[-1], tpr[-1] = 1, 1
        Auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)' % (Auc))
                # # 记录ROC曲线以及曲线下面积   
        f = open('experiments/img/roc_record01.txt', 'ab+')
        f.write(save_tag + '   AUC:' +  str(Auc) + '\n')
        f.write('FPR:' + str(list(fpr)) + '\n')
        f.write('TPR:' + str(list(tpr)) + '\n\n')
        f.close()

        # # #字典中的key值即为csv中列名
        # # dataframe = pd.DataFrame({'FPR':fpr,'TPR':tpr})
        # # #将DataFrame存储为csv,index表示是否显示行名，default=True
        # # dataframe.to_csv('experiments/img/roc_record.csv', index=False, sep=',')  
    else:
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in classes:
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
            fpr[i][0], tpr[i][0] = 0, 0
            fpr [i][-1], tpr[i][-1] = 1, 1
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in classes]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in classes:
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= len(classes)

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=2)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=2)

        colors = itertools.cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(classes, colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=1.5,
                     label='ROC of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))
            
            # # 记录ROC曲线以及曲线下面积   
            f = open('experiments/img/roc_record.txt', 'ab+')
            f.write(save_tag + '  AUC of class {0}:{1:f}\n'.format(i, roc_auc[i]))
            f.write('FPR:' + str(list(fpr[i])) + '\n')
            f.write('TPR:' + str(list(tpr[i])) + '\n\n')
            f.close()
        
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color=(0.6, 0.6, 0.6), alpha=.8)
    plt.xlim([-0.05, 1.05])  
    plt.ylim([-0.05, 1.05])  
    plt.xlabel('False Positive Rate')  
    plt.ylabel('True Positive Rate')  
    plt.title('Receiver operating curve')  
    plt.legend(loc="lower right") 
    plt.savefig('experiments/img/'+ save_tag + '_roc.png')
    plt.close('all') # 关闭图    
    
def accuracy(y_true, y_pred, classes, isPlot, save_tag = ''):
    # 计算混淆矩阵
    y = np.zeros(len(y_true))
    y_ = np.zeros(len(y_true))    
    for i in range(len(y_true)): 
        y[i] = np.argmax(y_true[i,:])
        y_[i] = np.argmax(y_pred[i,:])
    cnf_mat = confusion_matrix(y, y_)
    print cnf_mat
    
    if isPlot:
        # # 绘制混淆矩阵
        plot_confusion_matrix(cnf_mat, range(classes), save_tag=save_tag)
        # # 绘制ROC曲线
        plot_roc_curve(y_true, y_pred, range(classes), save_tag)

    if classes > 2: 
        # 计算多分类评价值
        Sens = recall_score(y, y_, average='macro')
        Prec = precision_score(y, y_, average='macro')
        F1 = f1_score(y, y_, average='weighted') 
        Support = precision_recall_fscore_support(y, y_, beta=0.5, average=None)
        print Support
        return Sens, Prec, F1, cnf_mat
    else:
        Acc = 1.0*(cnf_mat[1][1]+cnf_mat[0][0])/len(y_true)
        Sens = 1.0*cnf_mat[1][1]/(cnf_mat[1][1]+cnf_mat[1][0])
        Spec = 1.0*cnf_mat[0][0]/(cnf_mat[0][0]+cnf_mat[0][1])
        # 计算AUC值
        Auc = roc_auc_score(y_true[:,1], y_pred[:,1])
        return Acc, Sens, Spec, Auc 



        
# def create_record(path, num_classes):
    # '''
    # 此处我加载的数据目录如下：
    # 0 -- img1.jpg
         # img2.jpg
         # img3.jpg
         # ...
    # 1 -- img1.jpg
         # img2.jpg
         # ...
    # 2 -- ...
    # ...
    # '''
    # writer = tf.python_io.TFRecordWriter(path + ".tfrecords")
    # for index in range(num_classes):
        # class_path = path + "/" + str(index) + "/"
        # for img_name in os.listdir(class_path):
            # img_path = class_path + img_name
            # img = Image.open(img_path)
            # # img = img.resize((64, 64))
            # img_raw = img.tobytes() #将图片转化为原生bytes
            # example = tf.train.Example(features=tf.train.Features(feature={
                # "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                # 'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            # }))
            # writer.write(example.SerializeToString())
    # writer.close()