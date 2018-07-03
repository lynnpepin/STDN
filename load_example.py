import models
import numpy as np
import sys
import keras
from attention import Attention

####
# How to save a model:
####
'''
# yaml
yaml = model.to_yaml()
with open("stdn.yaml","w") as f:
    f.write(yaml)
# json
json = model.to_json()
with open("stdn.json","w") as f:
    f.write(json)
'''

####
# Load architecture from json, yaml, or (TODO) serialized object
####
'''
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
'''
# Serialized (TODO)

####
#Sanity check:
####
#From model.summary()
# Total params: 9,446,274
# Trainable params: 9,446,274
# Non-trainable params: 0

# Oh no. It looks like the Attention layers are erroneously saving/loading with 0 parameters
# 

####
# Load weights from file (TODO)
####
# model.load_weights("stdn_weights.h5")

sys.getsizeof(model)
