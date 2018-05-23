# -*- coding: UTF-8 -*-
from data_loader.liver_data_loader import LiverDataLoader
from models.liver_model import LiverModel
from trainers.liver_trainer import LiverModelTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args, timer
import numpy as np

S = [ # 融合方案
# ['E5', 'F5', 'G5', 'H5', 'I5', 'J5'],

['E5', 'F5', 'G5', 'H5', 'I5']

]

# 10次实验统计平均值
Acc = []
Sens = []
Prec = []
F1 = [] 
        
#OneVsAll 多分类对于每类而言
Mul_acc = []
Mul_sens = []
Mul_spec = []
Mul_auc = []


@timer
def main():
    # 获取配置文件路径
    # 运行：python main.py -c configs/fusion_config.json  #for MCF-3D-CNN
    #   Or: python main.py -c configs/3dcnn_config.json   #for 3DCNN
    # 可视化: tensorboard --logdir=experiments/2018-04-23/MCF-3D CNN/logs
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    create_dirs([config.tensorboard_log_dir, config.checkpoint_dir])
    
    for fusion_type in ['concat']:
        for Fusion in S:
            No = -1
            max_score = 0
            max_sens = 0
            max_prec = 0
            max_f1 = 0
            save_tag = str(Fusion) + fusion_type + '_012'
            #重复10次实验
            for i in range(config.repeat): 
                # 载入数据
                print('Create the data generator.')
                data_loader = LiverDataLoader(config)
                
                # 建立模型
                print('Create the model.')
                model = LiverModel(config, fusion_type, Fusion)

                # 训练模型、评估模型
                print('Create the trainer')
                trainer = LiverModelTrainer(model.model, data_loader.get_data(Fusion), config)
                print('Start training the model.')
                trainer.train(fusion_type, Fusion, i, max_score)
                
                score, sens, prec, f1 = trainer.getResults('avg')
                
                # Record the results
                Acc.append(score) 
                Sens.append(sens)   
                Prec.append(prec)
                F1.append(f1)
                
                mul_acc, mul_sens, mul_spec, mul_auc = trainer.getResults('mul')
                #OneVsAll
                Mul_acc.append(mul_acc)
                Mul_sens.append(mul_sens)
                Mul_spec.append(mul_spec)
                Mul_auc.append(mul_auc)
                
                # Record best result                
                if score > max_score:
                    No = i
                    max_score = score
                    max_sens = sens
                    max_prec = prec
                    max_f1 = f1
                
            # Save Overall
            fp = open('experiments/results.txt', 'ab+')
            fp.write(save_tag + '\nAvg @ Acc:%.4f+-%.4f Sens:%.4f+-%.4f Prec:%.4f+-%.4f F1:%.4f+-%.4f\n'\
                        %(np.mean(Acc), np.std(Acc), np.mean(Sens), np.std(Sens),
                        np.mean(Prec), np.std(Prec), np.mean(F1), np.std(F1)))
            print max_score, max_sens, max_prec, max_f1
            fp.write('Best@No%d  Acc:%.4f Sens:%.4f Prec:%.4f F1:%.4f\n\n'%(No, max_score, max_sens, max_prec, max_f1))                                             
            fp.close()
                
            #Save OneVsAll
            f = open('experiments/img/oneVsAll.txt', 'ab+')
            Acc_means, Acc_stds = np.mean(Mul_acc, 0), np.std(Mul_acc, 0)
            f.write('\nAcc: %.4f+-%.4f\t'%(Acc_means[0], Acc_stds[0]))
            f.write('%.4f+-%.4f\t'%(Acc_means[1], Acc_stds[1]))
            f.write('%.4f+-%.4f\t'%(Acc_means[2], Acc_stds[2]))
                
            Sens_means, Sens_stds = np.mean(Mul_sens, 0), np.std(Mul_sens, 0)
            f.write('\nSens: %.4f+-%.4f\t'%(Sens_means[0], Sens_stds[0]))
            f.write('%.4f+-%.4f\t'%(Sens_means[1], Sens_stds[1]))
            f.write('%.4f+-%.4f\t'%(Sens_means[2], Sens_stds[2]))
                
            Spec_means, Spec_stds = np.mean(Mul_spec, 0), np.std(Mul_spec, 0)
            f.write('\nSpec: %.4f+-%.4f\t'%(Spec_means[0], Spec_stds[0]))
            f.write('%.4f+-%.4f\t'%(Spec_means[1], Spec_stds[1]))
            f.write('%.4f+-%.4f\t'%(Spec_means[2], Spec_stds[2]))
                
            AUC_means, AUC_stds = np.mean(Mul_auc, 0), np.std(Mul_auc, 0)
            f.write('\nAUC: %.4f+-%.4f\t'%(AUC_means[0], AUC_stds[0]))
            f.write('%.4f+-%.4f\t'%(AUC_means[1], AUC_stds[1]))
            f.write('%.4f+-%.4f\t'%(AUC_means[2], AUC_stds[2]))
            f.close()

if __name__ == '__main__':
    main()