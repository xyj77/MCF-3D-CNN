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
        
        # 10次实验统计平均值
        self.Acc = []
        self.Sens = []
        self.Prec = []
        self.F1 = [] 

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

    def train(self, fusion_type, Fusion):
        max_score = 0
        save_tag = str(Fusion) + fusion_type + '_012'
        for i in range(self.config.repeat):
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
            sens, prec, f1, _ = accuracy(testLabels, pred_test, self.config.classes, 0)
            print sens, prec, f1    
                
            if max_score-score[1] < 0.03:
                self.Acc.append(score[1]) 
                self.Sens.append(sens)   
                self.Prec.append(prec)
                self.F1.append(f1)   
            if score[1] > max_score:
                max_score = score[1]
                #保存best_model  
                json_string = model.to_json() 
                open('experiments/models/' + save_tag + '_architecture.json','w').write(json_string)  
                model.save_weights('experiments/models/' + save_tag + '_weights.h5')
                #保存训练曲线和ROC曲线
                accuracy_curve(result, self.config.data_path, 1, save_tag)
                # accuracy(testLabels, pred_test, self.config.classes, 1, save_tag=save_tag)
                 
        fp = open('experiments/results.txt', 'ab+')
        fp.write(save_tag + '\nAvg @ Acc:%.4f+-%.4f Sens:%.4f+-%.4f Prec:%.4f+-%.4f F1:%.4f+-%.4f\n'\
                    %(np.mean(self.Acc), np.std(self.Acc), np.mean(self.Sens), np.std(self.Sens),
                    np.mean(self.Prec), np.std(self.Prec), np.mean(self.F1), np.std(self.F1)))
            
        model.load_weights('experiments/models/' + save_tag + '_weights.h5')
        score = model.evaluate(Test_list, testLabels, batch_size=1)
        print score
        pred_test = model.predict(Test_list)
        sens, prec, f1, cnf_mat = accuracy(testLabels, pred_test, self.config.classes, 1, save_tag=save_tag)
        fp.write('Best@ Acc:%.4f+-%.4f Sens:%.4f+-%.4f Prec:%.4f+-%.4f F1:%.4f+-%.4f\n'\
                    %(score[1], np.std(self.Acc), sens, np.std(self.Sens), prec, np.std(self.Prec), f1, np.std(self.F1)))
        fp.write(str(cnf_mat) + '\n\n')                                              
        fp.close()
        
        self.loss.extend(result.history['loss'])
        self.acc.extend(result.history['acc'])
        self.val_loss.extend(result.history['val_loss'])
        self.val_acc.extend(result.history['val_acc'])