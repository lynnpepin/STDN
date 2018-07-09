import argparse

import models
import file_loader
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error
import sys
# import xgboost as xgb
import json
import gc
import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session

from keras import backend as K
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Activation, concatenate, Input, Conv2D, Reshape, Flatten, Dropout, BatchNormalization, Concatenate
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
import datetime
import ipdb
import gc
import attention
from attention import Attention


class CustomStopper(keras.callbacks.EarlyStopping):
    def __init__(self,
                 monitor    = 'val_loss',
                 min_delta  = 0,
                 patience   = 0,
                 verbose    = 0,
                 mode       = 'auto',
                 start_epoch= 40
                ):
        super().__init__(monitor    = monitor,
                         min_delta  = min_delta,
                         patience   = patience,
                         verbose    = verbose,
                         mode       = mode
                        )
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)

def eval_together(y, pred_y, threshold):
    mask = y > threshold
    if np.sum(mask)==0:
        return -1
    mape = np.mean(np.abs(y[mask]-pred_y[mask])/y[mask])
    rmse = np.sqrt(np.mean(np.square(y[mask]-pred_y[mask])))

    return rmse, mape

def eval_lstm(y, pred_y, threshold):
    pickup_y        = y[:, 0]
    dropoff_y       = y[:, 1]
    pickup_pred_y   = pred_y[:, 0]
    dropoff_pred_y  = pred_y[:, 1]
    pickup_mask     = pickup_y > threshold
    dropoff_mask    = dropoff_y > threshold
    #pickup part
    if np.sum(pickup_mask)!=0:
        avg_pickup_mape = np.mean(np.abs(pickup_y[pickup_mask]-pickup_pred_y[pickup_mask])/pickup_y[pickup_mask])
        avg_pickup_rmse = np.sqrt(np.mean(np.square(pickup_y[pickup_mask]-pickup_pred_y[pickup_mask])))
    #dropoff part
    if np.sum(dropoff_mask)!=0:
        avg_dropoff_mape = np.mean(np.abs(dropoff_y[dropoff_mask]-dropoff_pred_y[dropoff_mask])/dropoff_y[dropoff_mask])
        avg_dropoff_rmse = np.sqrt(np.mean(np.square(dropoff_y[dropoff_mask]-dropoff_pred_y[dropoff_mask])))

    return (avg_pickup_rmse, avg_pickup_mape), (avg_dropoff_rmse, avg_dropoff_mape)

def print_time():
    print("Timestamp:", datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S"))

def main(
        att_lstm_num            = 3,
        long_term_lstm_seq_len  = 3,
        short_term_lstm_seq_len = 7,
        cnn_nbhd_size           = 3,
        nbhd_size               = 2,
        cnn_flat_size           = 128,
        batch_size              = 64,
        max_epochs              = 100,
        validation_split        = 0.2,
        early_stop              = EarlyStopping(),
        model_filename          = None,
        train_dataset           = "train",
        test_dataset            = "test",
        save_filename           = None
    ):
    """
    Samples raw data (from /data/*.npz files) to create training data for model,
    defines and compiles model,
    trains model,
    then saves it.
    """
    model_hdf5_path = "./hdf5s/"
    sampler = file_loader.file_loader()
    modeler = models.models()

    #training
    # Step 1. Create training dataset from raw data
    if V: print("Sampling data.")
    att_cnnx, att_flow, att_x, cnnx, flow, x, y = \
            sampler.sample_stdn(datatype                = train_dataset,
                                att_lstm_num            = att_lstm_num,
                                long_term_lstm_seq_len  = long_term_lstm_seq_len,
                                short_term_lstm_seq_len = short_term_lstm_seq_len,
                                nbhd_size               = nbhd_size,
                                cnn_nbhd_size           = cnn_nbhd_size)
    print_time()
    
    
    # Step 2. Compile model architecture
    if V: print("Creating model with input shape {1} / {0}".format(x.shape, cnnx[0].shape))

    model = modeler.stdn(att_lstm_num     = att_lstm_num,           #3
                         att_lstm_seq_len = long_term_lstm_seq_len, #3
                         lstm_seq_len     = len(cnnx),              #7
                         feature_vec_len  = x.shape[-1],            #160
                         cnn_flat_size    = cnn_flat_size,          #128
                         nbhd_size        = cnnx[0].shape[1],       #7
                         nbhd_type        = cnnx[0].shape[-1])      #2
    
    if V: print("\nModel created.")
    if V: print_time()
    if model_filename:
        if V: print("  Loading model weights from", model_filename)
        model.load_weights(model_hdf5_path + model_filename)
        if V: print_time()
    
    if V: print("Start training with input shape {1} / {0}\n".format( x.shape, cnnx[0].shape))
    # Step 3. Train model
    # Note for 'x=...' below:
    # 1.'+' is concat operator for list,
    # 2. Input() layers each take a section of this list; the format is implicit!
    # (See model architecture - there are multiple, different input layers.)
    model.fit(x                = att_cnnx + att_flow + att_x + cnnx + flow + [x,],
              y                = y,
              batch_size       = batch_size,
              validation_split = validation_split,
              epochs           = max_epochs,
              callbacks        = [early_stop]
             )
    if V: print("\nModel fit complete. Starting sampling of test data for evaluation.")
    if V: print_time()
    
    # Step 4. Test model against 'test' dataset.
    att_cnnx, att_flow, att_x, cnnx, flow, x, y = sampler.sample_stdn(datatype      = test_dataset,
                                                                      nbhd_size     = nbhd_size,
                                                                      cnn_nbhd_size = cnn_nbhd_size)
    if V: print_time()
    if V: print("Starting evaluation.")
    y_pred = model.predict(x = att_cnnx + att_flow + att_x + cnnx + flow + [x,],)
    threshold = float(sampler.threshold) / sampler.config["volume_train_max"]
    print("  Evaluating threshold: {0}.".format(threshold))
    (prmse, pmape), (drmse, dmape) = eval_lstm(y, y_pred, threshold)
    print("  Test on model:\npickup rmse = {0}, pickup mape = {1}%\n  dropoff rmse = {2}, dropoff mape = {3}%".format(prmse, pmape*100, drmse, dmape*100))
    if V: print("\nScore:", model.evaluate(att_cnnx + att_flow + att_x + cnnx + flow + [x,], y))
    
    if V: print("\nEvaluation finished. Saving model weights.")
    if V: print_time()
    # Step 5: Save model
    # This method replaces the .save()]
    if save_filename is None:
        currTime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        save_filename = currTime + "_weights.hdf5"
    
    model.save_weights(model_hdf5_path + save_filename)
    if V: print("Model weights saved to " + model_hdf5_path + save_filename , sep='')
    return

stop = CustomStopper(monitor    = 'val_loss',
                     min_delta  = 0,
                     patience   = 5,
                     verbose    = 0,
                     mode       = 'min',
                     start_epoch= 1)
                     #start_epoch= 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STDN with CLI")
    parser.add_argument("--model", "-m",
                        help="Load model weights from file under hdf5s/",
                        type=str, nargs=1)
    parser.add_argument("--epochs", "-e",
                        help="Amount of epochs to run (default 1).",
                        type=int, nargs=1)
    parser.add_argument("--batch", "-b",
                        help="Batch size (default 64).",
                        type=int, nargs=1)
    parser.add_argument("--verbose", "-v",
                        help="",
                        action="store_true")
    parser.add_argument("--train",
                        help="Name of training dataset to use. ('train', 'test', 'tiny', or 'tiny2')",
                        type=str, nargs=1)
    parser.add_argument("--test",
                        help="Name of testing dataset to use. ('train', 'test', 'tiny', or 'tiny2')",
                        type=str, nargs=1)
    parser.add_argument("--save", "-s",
                        help="Location to save model weights to in hdf5s/",
                        type=str, nargs=1)
    parser.add_argument("--gpunum", "-g",
                        help="GPU number. (Legacy(?))",
                        type=int, nargs=1)
    
    args = parser.parse_args()
    
    V = args.verbose
    
    model_filename = None if args.model is None else args.model[0]
    if V: print("  Model name:",model_filename)
    
    max_epochs = 1 if args.epochs is None else args.epochs[0]
    if V: print("  Max epochs:", max_epochs)
    
    batch_size = 64 if args.batch is None else args.batch[0]
    if V: print("  Batch size:", batch_size)
    
    train_data = "train" if args.train is None else args.train[0]
    test_data = "test" if args.test is None else args.test[0]
    if V: print("  Training on",train_data,"and testing on", test_data)
    
    save_filename = None if args.save is None else args.save[0]
    if V: print("  Save name:", save_filename)
    
    gpu_num = None if args.gpunum is None else args.gpunum[0]
    if V: print("  GPU number:", gpu_num)
    
    if gpu_num is not None:
        if gpu_num != "0" and gpu_num != "1":
            print("\nError gpu code {0}, use default.".format(gpu_num))
            gpu_num = "0"
        import os
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        ## Issue #152? Is that Keras or Tensorflow? I think it's Tensorflow
        # https://github.com/tensorflow/tensorflow/issues/152
        # https://github.com/keras-team/keras/issues/152
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
        print("\n***** Selecting gpu {0}".format(gpu_num))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # allow_growth - see https://www.tensorflow.org/programmers_guide/using_gpu
    set_session(tf.Session(config=config))

    
    if V: print("\n\n####################")
    if V: print("Starting main.py main()")
    if V: print_time()
    if V: print("####################\n")
    main(batch_size= batch_size,
         max_epochs= max_epochs,
         early_stop= stop,
         model_filename          = model_filename,
         train_dataset           = train_data,
         test_dataset            = test_data,
         save_filename           = save_filename
        )
    if V: print("\n####################")
    if V: print("All done!")
    if V: print_time()
    if V: print("####################\n\n")
