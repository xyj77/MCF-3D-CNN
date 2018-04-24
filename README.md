# Temporal-spatial Feature Learning of DCE-MR Images via 3DCNN
Keras code for paper: [Temporal-spatial Feature Learning of Dynamic Contrast Enhanced-MR Images via 3D Convolutional Neural Networks]

## Requirements

Python 2.7

TensorFlow

Keras

## Data
Data belongs to Beijing Friendship Hospital.

<div align="center">

<img align="center" width="600" src="https://github.com/xyj77/MCF-3D-CNN/raw/master/figures/DCE-MRI.PNG">

</div>


## To run the demo project:
1. Start the training using:
```shell
python main.py -c configs/fusion_config.json
```
2. Start Tensorboard visualization using:
```shell
tensorboard --logdir=experiments/Year-Month-Day/MCF-3D CNN/logs
```

## Tensor-based data representation
<div align="center">

<img align="center" width="600" src="https://github.com/xyj77/MCF-3D-CNN/raw/master/figures/TTD.png">

</div>




## Citing

If you use this code in your research, please use the following BibTeX entry.

```
@inproceedings{,
  title={},
  author={},
  booktitle={},
  year={}
}
```

## Reference

[Keras-Project-Template](https://github.com/Ahmkel/Keras-Project-Template)
