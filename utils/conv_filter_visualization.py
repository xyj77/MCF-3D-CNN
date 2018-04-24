#-*- coding:utf-8 -*-
from __future__ import print_function
import os, cv2
import scipy.io as sio
from scipy.misc import imsave
from sklearn.metrics import roc_auc_score, roc_curve, auc, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from PIL import Image
import numpy as np
from keras import backend as K
from keras.utils import np_utils 
from keras.models import model_from_json, Model
import tensorflow as tf

(w_in, h_in, s_in, c_in) = (32, 32, 5, 5)
margin = 5
samples_n = 24
samples = range(0, samples_n)

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x
def loadSplitData(path, classes):  
    train_data = []  
    train_labels = []  
    fp = open(os.path.join(path, "visual_train.txt"), 'r')
    # fp = open(os.path.join(path, "train.txt"), 'r')
    line = fp.readline()   # 调用文件的 readline()  
    while len(line):
        # print '*'*10,line      
        Level = int(line[0])
        imgpath = line[2:-1]
        mat = sio.loadmat(imgpath)
        # print mat.keys()
        train_data.append(mat['P'])
        train_labels.append(Level)
        line = fp.readline()
    fp.close()
    train_labels = np_utils.to_categorical(train_labels, classes)
    train_data = np.asarray(train_data, dtype="float32") 

    X_train, y_train = train_data, train_labels  

    test_data = []  
    test_labels = []  
    fp = open(os.path.join(path, "visual_test.txt"), 'r')
    # fp = open(os.path.join(path, "test.txt"), 'r')
    line = fp.readline()   # 调用文件的 readline()  
    while len(line):
        # print '*'*10,line      
        Level = int(line[0])
        imgpath = line[2:-1]
        mat = sio.loadmat(imgpath)
        # print mat.keys()
        test_data.append(mat['P'])
        test_labels.append(Level)
        line = fp.readline()
    fp.close()
    test_labels = np_utils.to_categorical(test_labels, classes)
    test_data = np.asarray(test_data, dtype="float32")

    X_test, y_test = test_data, test_labels
    
    return (X_train, y_train), (X_test, y_test)   
def combData(path, C, dict, isTrans):
    (trainData, trainLabels), (testData, testLabels) = loadSplitData(path, 3)
    classes = 3
    Train_list = []
    Test_list = []    
    for modual in C: 
        if len(modual) == 1 or len(modual) == 3:  
            input_shape = (trainData.shape[1], trainData.shape[2])
            Train_list.append(trainData[:,:,:,dict[modual]].reshape(trainData.shape[0], input_shape[0], input_shape[1]))
            Test_list.append(testData[:,:,:,dict[modual]].reshape(testData.shape[0], input_shape[0], input_shape[1])) 
        else:  
            input_shape = (trainData.shape[1], trainData.shape[2], 5)
            Train_list.append(trainData[:,:,:,dict[modual]].reshape(trainData.shape[0], input_shape[0], input_shape[1], 5))
            Test_list.append(testData[:,:,:,dict[modual]].reshape(testData.shape[0], input_shape[0], input_shape[1], 5))
       
    if len(C[0]) == 1 or len(C[0]) == 3:         
        X_train = tf.cast(tf.stack(Train_list, axis=3), tf.float32)
        X_test = tf.cast(tf.stack(Test_list, axis=3), tf.float32)
    else:
        X_train = tf.cast(tf.stack(Train_list, axis=4), tf.float32)
        X_test = tf.cast(tf.stack(Test_list, axis=4), tf.float32)
        if isTrans:
            # # 转置
            X_train = tf.transpose(X_train, perm=[0,1,2,4,3])
            X_test = tf.transpose(X_test, perm=[0,1,2,4,3])
  
    with tf.Session() as sess:
        X_train, X_test = X_train.eval(), X_test.eval()        
    return (X_train, trainLabels), (X_test, testLabels), classes
def accuracy(y_true, y_pred):        
    # 计算混淆矩阵
    y = np.zeros(len(y_true))
    y_ = np.zeros(len(y_true))    
    for i in range(len(y_true)): 
        y[i] = np.argmax(y_true[i,:])
        y_[i] = np.argmax(y_pred[i,:])
    cnf_mat = confusion_matrix(y, y_)
    
    # Acc = 1.0*(cnf_mat[1][1]+cnf_mat[0][0])/len(y_true)
    # Sens = 1.0*cnf_mat[1][1]/(cnf_mat[1][1]+cnf_mat[1][0])
    # Spec = 1.0*cnf_mat[0][0]/(cnf_mat[0][0]+cnf_mat[0][1])
    
    # # 绘制ROC曲线
    # fpr, tpr, thresholds = roc_curve(y_true[:,0], y_pred[:,0])
    # Auc = auc(fpr, tpr)
    
    
    # 计算多分类评价值
    Sens = recall_score(y, y_, average='macro')
    Prec = precision_score(y, y_, average='macro')
    F1 = f1_score(y, y_, average='weighted') 
    Support = precision_recall_fscore_support(y, y_, beta=0.5, average=None)
    return Sens, Prec, F1, cnf_mat
def save_map_img(layer_name, ex_model, (map_w, map_h, map_s), map_n):
    width = map_w * map_n + (map_n-1) * margin
    height = map_h * len(samples) + margin * (len(samples) - 1)
    if map_s == 1 or map_s == 3:
        stitched_filters = np.zeros((width, height, map_s))
    else:
        raise RuntimeError('Channel Error')
    for j, n in enumerate(samples):
        tr_features = ex_model.predict(trainData[n,:,:,:,:].reshape(1,w_in, h_in, s_in, c_in))
        feature_map = deprocess_image(tr_features)
        # fill the picture with our saved filters
        for i in range(map_n):
            stitched_filters[(map_w+margin)*i : (map_w+margin)*i+map_w,
                             (map_h+margin)*j : (map_h+margin)*j+map_h, :] = feature_map[0,:,:,:,i]     
    # save the result to disk
    # imsave(layer_name+'feature.png', stitched_filters)
    cv2.imwrite('../experiments/img/'+layer_name+'feature.png', stitched_filters)


# build the VGG16 network with ImageNet weights
model = model_from_json(open('../experiments/models/[\'E5\', \'F5\', \'G5\', \'H5\', \'I5\']_1_architecture.json').read())  
model.load_weights('../experiments/models/[\'E5\', \'F5\', \'G5\', \'H5\', \'I5\']_1_weights.h5')
print('Model loaded.')
model.summary()
# # get the symbolic outputs of each "key" layer (we gave them unique names).
# layer_dict = dict([(layer.name, layer) for layer in model.layers[0:]])
# for layer in model.layers[0:]:
   # print(layer)

#读取数据
path = '../data_loader/ABKEFGHIJ_EFGHIJ_FGHIJ-E_FGHIJ5-E'
dict = {'A': 0, 'B': 1, 'K': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'J': 8, \
        'E5':range(9,14), 'F5':range(14,19), 'G5':range(19,24), 'H5':range(24,29), 'I5':range(29,34), 'J5':range(34,39), \
        'F-E':39, 'G-E':40, 'H-E':41, 'I-E':42, 'J-E':43, \
        'F-E5':range(44,49), 'G-E5':range(49,54), 'H-E5':range(54,59), 'I-E5':range(59,64), 'J-E5':range(64,69)}
C = ['E5', 'F5', 'G5', 'H5', 'I5']
(trainData, trainLabels), (testData, testLabels), classes = combData(path, C, dict, 1)
print(trainLabels[:samples_n, :])

#验证模型
pred_test = model.predict(trainData)
sens, prec, f1, cnf_mat = accuracy(testLabels, pred_test)
print(sens, prec, f1, '\n', cnf_mat)

# the name of the layer we want to visualize
#绘制第二个卷积层特征图
layer_name = 'conv1'
ex_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
#绘制图像 
save_map_img(layer_name, ex_model, (30, 30, 3), 6)


#绘制第二个卷积层特征图
layer_name = 'conv2'
ex_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
save_map_img(layer_name, ex_model, (12, 12, 1), 8)