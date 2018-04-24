#-*- coding:utf-8 -*-
from __future__ import print_function

from scipy.misc import imsave
import numpy as np
from keras.models import model_from_json
margin = 1

def save_kernel_img(layer_name, kernel, (k_w, k_h, k_s), (n_in, k_n)):
    width = k_w * k_s * k_n + (k_s * k_n - 1) * margin
    height = k_h * n_in + (n_in - 1) * margin
    stitched_filters = np.ones((width, height))
    # fill the picture with our saved filters
    for k in range(k_n):
        for j in range(n_in):
            for i in range(k_s):
                offset = (k_w + margin) * k_s
                stitched_filters[(k_w+margin)*i + k*offset : (k_w+margin)*i + k_w+k*offset,
                                 (k_h+margin)*j : (k_h+margin)*j+k_h] = kernel[:,:,i,j,k]
    # save the result to disk
    imsave('img/'+layer_name+'kernels.png', stitched_filters)    


#载入模型
model = model_from_json(open('results/Trans_models20180302/[\'E5\', \'F5\', \'G5\', \'H5\', \'I5\']_1_architecture.json').read())  
model.load_weights('results/Trans_models20180302/[\'E5\', \'F5\', \'G5\', \'H5\', \'I5\']_1_weights.h5')
print('Model loaded.')
model.summary()
# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers[0:]])
#for layer in model.layers[0:]:
#    print(layer)


# the name of the layer we want to visualize
layer_name = 'conv1'
kernel = layer_dict[layer_name].get_weights()[0]
save_kernel_img(layer_name, kernel, (3, 3, 3), (5, 6))

#可视化第二层卷积核
layer_name = 'conv2'
kernel = layer_dict[layer_name].get_weights()[0]
save_kernel_img(layer_name, kernel, (4, 4, 3), (6, 8))