# -*- coding: UTF-8 -*- 
from base.base_model import BaseModel
import sys

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Merge
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, Adam
from keras.utils.vis_utils import plot_model

class LiverModel(BaseModel):
    def __init__(self, config, fusion_type, Fusion):
        super(LiverModel, self).__init__(config)
        self.model_type = config.exp_name
        if self.model_type == 'MCF-3D-CNN':
            self.model = self.build_fusion_model(fusion_type, Fusion)
        elif self.model_type == '3DCNN':
            self.model = self.build_3dcnn_model(fusion_type, Fusion)

    def build_fusion_model(self, fusion_type, Fusion):
        model_list = []
        for modual in Fusion:
            if len(modual) == 1: 
                input_shape = (32, 32, 1)
                single_model = self.cnn_2D(input_shape, modual) 
            else:
                input_shape = (32, 32, 5, 1)
                single_model = self.cnn_3D(input_shape, modual) 
                  
            model_list.append(single_model)
        
        # 融合模型
        model = self.nn_fusion(model_list, self.config.classes, fusion_type)
        # 统计参数
        model.summary()
        plot_model(model,to_file='experiments/img/' + str(Fusion) + fusion_type + r'_model.png',show_shapes=True)
        print '    Saving model  Architecture'
        
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        # model.compile(optimizer=adam, loss=self.mycrossentropy, metrics=['accuracy']) #有改善，但不稳定
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy']) 
        
        return model

    def build_3dcnn_model(self, fusion_type, Fusion):
        if len(Fusion[0]) == 1: 
            input_shape = (32, 32, len(Fusion))
            model = self.cnn_2D(input_shape) 
        else:
            input_shape = (32, 32, 5, len(Fusion))
            model = self.cnn_3D(input_shape) 
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu', name = 'fc2'))
        model.add(Dense(self.config.classes, activation='softmax', name = 'fc3')) 
        
        # 统计参数
        model.summary()
        plot_model(model,to_file='experiments/img/' + str(Fusion) + fusion_type + r'_model.png',show_shapes=True)
        print '    Saving model  Architecture'
        
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        # model.compile(optimizer=adam, loss=self.mycrossentropy, metrics=['accuracy']) #有改善，但不稳定
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy']) 
        
        return model        
        
    def cnn_2D(self, input_shape, modual=''):
        #建立Sequential模型    
        model = Sequential() 
        model.add(Conv2D(
                filters = 6,
                kernel_size = (3, 3),
                input_shape = input_shape,
                activation='relu',
                kernel_initializer='he_normal',
                name = modual+'conv1'
            ))# now 30x30x6
        model.add(MaxPooling2D(pool_size=(2,2)))# now 15x15x6
        model.add(Conv2D(
                filters = 8,
                kernel_size = (4, 4),
                activation='relu',
                kernel_initializer='he_normal',
                name = modual+'conv2'
            ))# now 12x12x8
        model.add(MaxPooling2D(pool_size=(2,2)))# now 6x6x8
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu', name = modual+'fc1'))
      
        return model

    def cnn_3D(self, input_shape, modual=''):
        #建立Sequential模型    
        model = Sequential() 
        model.add(Convolution3D(
                filters = 6,
                kernel_size = (3, 3, 3),
                input_shape = input_shape,
                activation='relu',
                kernel_initializer='he_normal',
                name = modual+'conv1'
            ))# now 30x30x3x6
        model.add(MaxPooling3D(pool_size=(2,2,1)))# now 15x15x3x6
        model.add(Convolution3D(
                filters = 8,
                kernel_size = (4, 4, 3),
                activation='relu',
                kernel_initializer='he_normal',
                name = modual+'conv2'
            ))# now 12x12x1x8
        model.add(MaxPooling3D(pool_size=(2,2,1)))# now 6x6x1x8
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu', name = modual+'fc1'))
      
        return model    

    def nn_fusion(self, model_list, classes, fusion_type):
        model = Sequential()
        model.add(Merge(model_list, mode=fusion_type))
        model.add(Dropout(0.5))
        
        if fusion_type in ['ave', 'sum', 'mul', 'dot', 'cos']:
            model.add(Dense(32,activation='relu'))
        else:
            model.add(Dense(128,activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(32,activation='relu'))
        model.add(Dense(classes, activation='softmax'))        
        return model
    
    # 定义损失函数        
    def mycrossentropy(self, y_true, y_pred):
        e = 0
        # for i in range(y_true.shape[0]):
            # for j in range(3):
                # sum += 0.1*(-1**y_true(i,j))*exp(abs(np.argmax(y_true[i,:])-j))*log(y_pred(i,j))
        # return sum/len

        # y = np.argmax(y_true, axis=1)
        # y_ = np.argmax(y_pred, axis=1)
        # print '*****************',y_pred
                
        # return (1-e)*K.categorical_crossentropy(y_pred,y_true) - e*K.categorical_crossentropy(y_pred, (1-y_true)/(classes-1)) 
        return (1-e)*K.categorical_crossentropy(y_pred,y_true) + e*K.categorical_crossentropy(y_pred, K.ones_like(y_pred)/2) 