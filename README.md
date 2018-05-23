# Temporal-spatial Feature Learning of DCE-MR Images via 3DCNN
Keras code for paper: [Temporal-spatial Feature Learning of Dynamic Contrast Enhanced-MR Images via 3D Convolutional Neural Networks]

## Requirements

Python 2.7

TensorFlow

Keras

## To run the demo project:
1. Start the training using:
```shell
python main.py -c configs/fusion_config.json  # MCF-3D-CNN
```
```shell
python main.py -c configs/3dcnn_config.json   # 3DCNN
```

2. Start Tensorboard visualization using:
```shell
tensorboard --logdir=experiments/Year-Month-Day/Ex-name/logs
```

## Data
Data belongs to Beijing Friendship Hospital.

<div align="center">

<img align="center" width="600" src="https://github.com/xyj77/MCF-3D-CNN/raw/master/figures/DCE-MRI.png">

</div>

## Tensor-based data representation
<div align="center">

<img align="center" width="600" src="https://github.com/xyj77/MCF-3D-CNN/raw/master/figures/TTD.png">

</div>

## Results
<div align="center">

**Tabel1 The results of discriminating the HCC and cirrhosis**
<img align="center" width="600" src="https://github.com/xyj77/MCF-3D-CNN/raw/master/figures/result1.png">

**Tabel2 The results of non-invasive assessment of HCC differentiation**
<img align="center" width="600" src="https://github.com/xyj77/MCF-3D-CNN/raw/master/figures/result2.png">
<img align="center" width="600" src="https://github.com/xyj77/MCF-3D-CNN/raw/master/figures/concat_roc.png">

</div>

## Feature maps of C1 and C2 convolution layer
<div align="center">

<img align="center" width="600" src="https://github.com/xyj77/MCF-3D-CNN/raw/master/figures/FeatureMaps.png">

</div>

## Reference

[Keras-Project-Template](https://github.com/Ahmkel/Keras-Project-Template)
