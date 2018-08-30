import numpy as np
from models import models
model = models().stdn(3, 3, 7, 160, nbhd_size = 7)
model.load_weights("./hdf5s/man_t0_040.h5")
from file_loader import file_loader
sampler = file_loader()
att_cnnx, att_flow, att_x, cnnx, flow, x, y = sampler.sample_stdn("test", 3, 3, 7, 7, 48, 2, 3)
# Replace dataset with "test" when proper!

# Get masked YP, YT
y_pred = model.predict(x = att_cnnx + att_flow + att_x + cnnx + flow + [x,],)
threshold = sampler.threshold / sampler.volume_max   
YP = y_pred.reshape(-1,10,20,2).swapaxes(1,2)
YT = y.reshape(-1,10,20,2).swapaxes(1,2)
YP[YT < threshold] = 0
YT[YT < threshold] = 0

# Calculating
N = np.count_nonzero(YT)
TN = YT.shape[0]            # Number of time slots
SN = np.prod(YT.shape[1:3]) # Number of grid areas
# Errors
E = np.abs(YT-YP)           #(-1, 20, 10, 2)
APE = np.nan_to_num(E / YT)
# Location and temporal E
SE = E.sum(axis=0)          #(20,10,2)
TE = E.sum(axis=(1,2))      #(-1,2)
# Location and temporal SE
SSE = (E**2).sum(axis=0)
TSE = (E**2).sum(axis=(1,2))

# Note: Divide by the proper N
# Location and temporal MAPE
SMAPE = APE.sum(axis=0)/TN
TMAPE = APE.sum(axis=(1,2))/SN
# Location and temporal RMSE
SRMSE = np.sqrt(SSE/TN)
TRMSE = np.sqrt(TSE/SN)

np.savez("errors.npz",SMAPE=SMAPE,TMAPE=TMAPE,SRMSE=SRMSE,TRMSE=TRMSE)

# TODO: Visualize, print
