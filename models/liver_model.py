# -*- coding: UTF-8 -*- 
from base.base_model import BaseModel
import sys

from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Concatenate, Input
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, Adam
from keras.utils.vis_utils import plot_model

import keras.backend as K 

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
        input_list = []
        for modual in Fusion:
            if len(modual) == 1: 
                input_shape = (32, 32, 1)
                signle_input,single_model = self.cnn_2D(input_shape, modual) 
            else:
                input_shape = (32, 32, 5, 1)
                signle_input,single_model = self.cnn_3D(input_shape, modual) 
                  
            model_list.append(single_model)
            input_list.append(signle_input)
        # 融合模型
        model = self.nn_fusion(input_list,model_list, self.config.classes, fusion_type)
        # 统计参数
        model.summary()
        plot_model(model,to_file='experiments/img/' + str(Fusion) + fusion_type + r'_model.png',show_shapes=True)
        print('    Saving model  Architecture')
        # raw_input()
        
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        # model.compile(optimizer=adam, loss=self.mycrossentropy, metrics=['accuracy']) #有改善，但不稳定
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy']) 
        
        return model

    def build_3dcnn_model(self, fusion_type, Fusion):
        if len(Fusion[0]) == 1: 
            input_shape = (32, 32, len(Fusion))
            model_in,model = self.cnn_2D(input_shape) 
        else:
            input_shape = (32, 32, 5, len(Fusion))
            model_in,model = self.cnn_3D(input_shape) 
        model = Dropout(0.5)(model)
        model = Dense(32, activation='relu', name = 'fc2')(model)
        model = Dense(self.config.classes, activation='softmax', name = 'fc3')(model) 
        model = Model(input=model_in,output=model)
        # 统计参数
        # model.summary()
        plot_model(model,to_file='experiments/img/' + str(Fusion) + fusion_type + r'_model.png',show_shapes=True)
        print('    Saving model  Architecture')
        
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        # model.compile(optimizer=adam, loss=self.mycrossentropy, metrics=['accuracy']) #有改善，但不稳定
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy']) 
        
        return model        
        
    def cnn_2D(self, input_shape, modual=''):
        #建立Sequential模型    
        model_in = Input(input_shape) 
        model = Conv2D(
                filters = 6,
                kernel_size = (3, 3),
                input_shape = input_shape,
                activation='relu',
                kernel_initializer='he_normal',
                name = modual+'conv1'
            )(model_in)# now 30x30x6
        model = MaxPooling2D(pool_size=(2,2))(model)# now 15x15x6
        model = Conv2D(
                filters = 8,
                kernel_size = (4, 4),
                activation='relu',
                kernel_initializer='he_normal',
                name = modual+'conv2'
            )(model)# now 12x12x8
        model = MaxPooling2D(pool_size=(2,2))(model)# now 6x6x8
        model = Flatten()(model)
        model = Dropout(0.5)(model)
        model_out = Dense(100, activation='relu', name = modual+'fc1')(model)
      
        return model_in, model_out

    def cnn_3D(self, input_shape, modual=''):
        #建立Sequential模型
        model_in = Input(input_shape)    
        model = Convolution3D(
                filters = 6,
                kernel_size = (3, 3, 3),
                input_shape = input_shape,
                activation='relu',
                kernel_initializer='he_normal',
                name = modual+'conv1'
            )(model_in)# now 30x30x3x6
        model = MaxPooling3D(pool_size=(2,2,1))(model)# now 15x15x3x6
        model = Convolution3D(
                filters = 8,
                kernel_size = (4, 4, 3),
                activation='relu',
                kernel_initializer='he_normal',
                name = modual+'conv2'
            )(model)# now 12x12x1x8
        model = MaxPooling3D(pool_size=(2,2,1))(model)# now 6x6x1x8
        model = Flatten()(model)
        model = Dropout(0.5)(model)
        model_out = Dense(100, activation='relu', name = modual+'fc1')(model)
      
        return model_in, model_out    

    def nn_fusion(self, input_list, model_list, classes, fusion_type):
        merged_model = Concatenate(axis=-1)(model_list)
        model = Dropout(0.5)(merged_model)
       
        if fusion_type in ['ave', 'sum', 'mul', 'dot', 'cos']:
            model = Dense(32,activation='relu')(model)
        else:
            model = Dense(128,activation='relu')(model)
            model = Dropout(0.5)(model)
            model = Dense(32,activation='relu')(model)
	    model = Dense(classes, activation='softmax')(model)   
	    model_out = Model(input=input_list,output=model)     
        return model_out
    
    # 定义损失函数        
    def mycrossentropy(self, y_true, y_pred):
        e = 0.3
        # for i in range(y_true.shape[0]):
            # for j in range(3):
                # sum += 0.1*(-1**y_true(i,j))*exp(abs(np.argmax(y_true[i,:])-j))*log(y_pred(i,j))
        # return sum/len

        # y = np.argmax(y_true, axis=1)
        # y_ = np.argmax(y_pred, axis=1)
        # print '*****************',y_pred
                
        # return (1-e)*K.categorical_crossentropy(y_pred,y_true) - e*K.categorical_crossentropy(y_pred, (1-y_true)/(self.config.classes-1)) 
        return (1-e)*K.categorical_crossentropy(y_pred,y_true) + e*K.categorical_crossentropy(y_pred, K.ones_like(y_pred)/2) 
