# -*- coding: UTF-8 -*-
from data_loader.liver_data_loader import LiverDataLoader
from models.liver_model import LiverModel
from trainers.liver_trainer import LiverModelTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args

S = [ # 融合方案
# ['E5', 'F5', 'G5', 'H5', 'I5', 'A'],

['E5', 'F5', 'G5', 'H5', 'I5']

]

def main():
    # 获取配置文件路径
    # 运行：python main.py -c configs/fusion_config.json
    #   Or: python main.py -c configs/3dcnn_config.json
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
            # 加载数据
            print('Create the data generator.')
            data_loader = LiverDataLoader(config)
            
            # 建立模型
            print('Create the model.')
            model = LiverModel(config, fusion_type, Fusion)

            # 训练模型、评估模型
            print('Create the trainer')
            trainer = LiverModelTrainer(model.model, data_loader.get_data(Fusion), config)
            print('Start training the model.')
            trainer.train(fusion_type, Fusion)

if __name__ == '__main__':
    main()
