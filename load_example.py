import models
import numpy as np
import sys
import keras
#from attention import Attention

# This code contains a number of example code snippets for loading/saving a model.
# It's not meant to be run as-is.

########
# WIP/TODO: The ideal way to save/load does not work
#   This section attempts to save to yaml/json, but fails to capture
#   vital information about the Attention layers.
########
'''
####
# How to save a model:
####
# yaml
yaml = model.to_yaml()
with open("stdn.yaml","w") as f:
    f.write(yaml)
# json
json = model.to_json()
with open("stdn.json","w") as f:
    f.write(json)

####
# Load architecture from json, yaml, or (TODO) serialized object
####

# json
with open("stdn.json") as f:
    json = f.read()
mjson = keras.models.model_from_json(json, custom_objects={'Attention':Attention(method='cba')})
#From summary:
# Total params: 9,347,586
# Trainable params: 9,347,586
# Non-trainable params: 0

# yaml
with open("stdn.yaml") as f:
    yaml = f.read()
myaml = keras.models.model_from_yaml(yaml, custom_objects={'Attention':Attention(method='cba')})
#From summary:
# Total params: 9,347,586
# Trainable params: 9,347,586
# Non-trainable params: 0

####
#Sanity check:
####
#From model.summary()
# Total params: 9,446,274
# Trainable params: 9,446,274
# Non-trainable params: 0
# Oh no. It looks like the Attention layers are erroneously saving/loading with 0 parameters
# 
'''


########
# Instantiate model architecture in code, and then load weights from file
# This method works and correctly re-creates the model.
########
# model.save_weights("stdn_weights.h5")

modeler = models.models()
new_m = modeler.stdn(att_lstm_num = 3,
                 att_lstm_seq_len = 3,
                 lstm_seq_len = 7,
                 feature_vec_len = 160,
                 cnn_flat_size = 128,
                 nbhd_size = 7,
                 nbhd_type = 2)

new_m.load_weights("stdn_weights.h5")

if __name__ == '__main__':
    print("This code is not meant to be run from main.")
    print("It's just a number of Python exmaple snippets for my notes.")
