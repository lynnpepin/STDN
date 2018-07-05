'''
This was the script used to generate volume_tiny.npz and flow_tiny.npz.

They are to be used just for testing changes to the model architecture, as they are small and
expected to fit within RAM.
'''

'''
Inspecting the data to determine what axis samples ar econtained on:

>>> data_train = np.load("data/volume_train.npz")
>>> data['volume'].shape
(960, 10, 20, 2)
>>> data_train['volume'].shape
(1920, 10, 20, 2)

#volume samples contained on axis 0
'''

'''
>>> flow_train = np.load("data/flow_train.npz")
>>> flow_data['flow'].shape
(2, 960, 10, 20, 10, 20)
>>> flow_train['flow'].shape
(2, 1920, 10, 20, 10, 20)

#flow samples contained on axis 1

'''

import numpy as np; from sys import getsizeof as sz
data = np.load("data/volume_test.npz")
flow_data = np.load("data/flow_test.npz")

volume = data['volume'][:500,:,:,:]
flow = flow_data['flow'][:,:500,:,:,:,:]

np.savez("data/volume_tiny.npz", volume)
np.savez("data/flow_tiny.npz", flow)

exit()

# Quick copy-paste for loading data, flow_data in interpreter
import numpy as np; from sys import getsizeof as sz
data = np.load("data/volume_tiny.npz")['arr_0']
flow_data = np.load("data/flow_tiny.npz")['arr_0']
