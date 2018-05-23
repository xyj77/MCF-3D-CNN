# -*- coding: UTF-8 -*- 
from base.base_trainer import BaseTrain
import sys
import os

from keras.models import model_from_json
from keras import callbacks
import math

import tensorflow as tf
from utils.utils import *

from keras.callbacks import ModelCheckpoint, TensorBoard

class LiverModelTrainer(BaseTrain):
    def __init__(self, model, data, config):
        super(LiverModelTrainer, self).__init__(model, data, config)
        self.model = model
        self.data = data        
        self.patience = config.patience
        self.min_lr = config.min_lr
        
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()
        
        self.trainOver = False
    
    def getResults(self, tag):
        if self.trainOver: #是否训练完成
            if tag == 'avg':
                return self.Acc, self.Sens, self.Prec, self.F1
            elif tag == 'mul':
                return self.Mul_acc, self.Mul_sens, self.Mul_spec, self.Mul_auc
        else:
            raise RuntimeError('Please train the model first!')

    def init_callbacks(self):
        # self.callbacks.append(
            # ModelCheckpoint(
                # filepath=os.path.join(self.config.checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp_name),
                # monitor=self.config.checkpoint_monitor,
                # mode=self.config.checkpoint_mode,
                # save_best_only=self.config.checkpoint_save_best_only,
                # save_weights_only=self.config.checkpoint_save_weights_only,
                # verbose=self.config.checkpoint_verbose,
            # )
        # )

        # self.callbacks.append(
                # TensorBoard(
                    # log_dir=self.config.tensorboard_log_dir,
                    # write_graph=self.config.tensorboard_write_graph,
                # )
            # )
        #学习率衰减
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1/math.e,
                                                verbose=1, patience=self.patience, min_lr=self.min_lr)
        self.callbacks.append(reduce_lr)  
        # if hasattr(self.config,"comet_api_key"):
            # from comet_ml import Experiment
            # experiment = Experiment(api_key=self.config.comet_api_key, project_name=self.config.exp_name)
            # experiment.disable_mp()
            # experiment.log_multiple_params(self.config)
            # self.callbacks.append(experiment.get_keras_callback())

    def train(self, fusion_type, Fusion, i, max_score):
        save_tag = str(Fusion) + fusion_type + '_012'
        Train_list = []
        Test_list = []
        (Train_list, trainLabels), (Test_list, testLabels) = (self.data[0], self.data[1]), (self.data[2], self.data[3])

        # 训练模型
        model = self.model          
        result = model.fit(Train_list, trainLabels, batch_size=self.config.batch_size, epochs=self.config.num_epochs, 
                            validation_data=(Test_list, testLabels), verbose=self.config.verbose_training, 
                            shuffle=True, callbacks= self.callbacks)
        #评估模型
        score = model.evaluate(Test_list, testLabels, batch_size=1)
        print '*'*20 + str(score)
        pred_test = model.predict(Test_list)
        sens, prec, f1, _ = cnf_roc(testLabels, pred_test, self.config.classes, 0)
        print sens, prec, f1 
        
        self.Acc = score[1]
        self.Sens = sens
        self.Prec = prec
        self.F1 = f1  

        if score[1] > max_score:
            #保存best_model          
            json_string = model.to_json() 
            open('experiments/models/' + save_tag + '_architecture.json','w').write(json_string)  
            model.save_weights('experiments/models/' + save_tag + '_weights.h5')
            #保存训练曲线和ROC曲线
            train_curve(result, self.config.data_path, 1, save_tag)
            cnf_roc(testLabels, pred_test, self.config.classes, 1, save_tag=save_tag)
        
        # 保存每次的roc和cnf  
        save_cnf_roc(testLabels, pred_test, self.config.classes, 1, save_tag=save_tag+'_'+str(i))
            
        #计算每个类别单独的敏感度特异度以及oneVsAll AUC
        one_Vs_all_acc, one_Vs_all_sens, one_Vs_all_spec, one_Vs_all_auc \
                    = oneVsAll(testLabels, pred_test, self.config.classes, save_tag+'_'+str(i))
        print one_Vs_all_acc, one_Vs_all_sens, one_Vs_all_spec, one_Vs_all_auc 

        #OneVsAll
        self.Mul_acc = one_Vs_all_acc
        self.Mul_sens = one_Vs_all_sens
        self.Mul_spec = one_Vs_all_spec
        self.Mul_auc = one_Vs_all_auc
        # print '**'*10
        # raw_input()
        
        #训练结束
        self.trainOver = True
        
        
        # self.loss.extend(result.history['loss'])
        # self.acc.extend(result.history['acc'])
        # self.val_loss.extend(result.history['val_loss'])
        # self.val_acc.extend(result.history['val_acc'])