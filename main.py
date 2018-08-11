import argparse
import models
import file_loader
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error
import sys
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
from math import floor

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


stop = CustomStopper(monitor    = 'val_loss',
                     min_delta  = 0,
                     patience   = 5,
                     verbose    = 0,
                     mode       = 'min',
                     start_epoch= 40)


capsstop = CustomStopper(monitor    = 'val_loss',
                     min_delta  = 0,
                     patience   = 15,
                     verbose    = 0,
                     mode       = 'min',
                     start_epoch= 150)

def eval_together(y, pred_y, threshold):
    # Not used
    mask = y > threshold
    if np.sum(mask)==0:
        return -1
    mape = np.mean(np.abs(y[mask]-pred_y[mask])/y[mask])
    rmse = np.sqrt(np.mean(np.square(y[mask]-pred_y[mask])))
    return rmse, mape

def eval_caps(y, pred_y, threshold):
    # TODO: This should be able to replace the eval_lstm
    # in a general manner by using this syntax:
    # pickup_y      = y[:, ..., 0]
    pickup_y        = y[:,:,:,0]
    dropoff_y       = y[:,:,:,1]
    pickup_pred_y   = pred_y[:,:,:,0]
    dropoff_pred_y  = pred_y[:,:,:,1]
    pickup_mask     = pickup_y > threshold
    dropoff_mask    = dropoff_y > threshold
    # pickup_y[pickup_mask] is a 1D array of length np.sum(pickup_mask)
    # TODO: Make sure this actually works correctly, too
    if np.sum(pickup_mask) != 0:
        avg_pickup_mape = np.mean(np.abs(pickup_y[pickup_mask]-pickup_pred_y[pickup_mask])/pickup_y[pickup_mask])
        avg_pickup_rmse = np.sqrt(np.mean(np.square(pickup_y[pickup_mask]-pickup_pred_y[pickup_mask])))
    if np.sum(dropoff_mask)!=0:
        avg_dropoff_mape = np.mean(np.abs(dropoff_y[dropoff_mask]-dropoff_pred_y[dropoff_mask])/dropoff_y[dropoff_mask])
        avg_dropoff_rmse = np.sqrt(np.mean(np.square(dropoff_y[dropoff_mask]-dropoff_pred_y[dropoff_mask])))
    return (avg_pickup_rmse, avg_pickup_mape), (avg_dropoff_rmse, avg_dropoff_mape)
        
def eval_lstm(y, pred_y, threshold):
    # For the STDN
    pickup_y        = y[:, 0]       # y, pred_y of shape (no of samples, 2)
    dropoff_y       = y[:, 1]       # These elements to the left are of shape(no of samples)
    pickup_pred_y   = pred_y[:, 0]
    dropoff_pred_y  = pred_y[:, 1]
    pickup_mask     = pickup_y > threshold      # Any element less than the threshhold gets set to 0
    dropoff_mask    = dropoff_y > threshold     # e.g. a = np.array([3,1,2,7]); mask = a > 2; a[mask] == array([3,7])
    #pickup part
    if np.sum(pickup_mask)!=0:
        # mean( |pickup_y - pickup_pred_y| /pickup_y); masked by the same mask
        avg_pickup_mape = np.mean(np.abs(pickup_y[pickup_mask]-pickup_pred_y[pickup_mask])/pickup_y[pickup_mask])
        # sqrt(mean(square(pickup_y - pickup_pred_y)))
        avg_pickup_rmse = np.sqrt(np.mean(np.square(pickup_y[pickup_mask]-pickup_pred_y[pickup_mask])))
    #dropoff part
    if np.sum(dropoff_mask)!=0:
        # Same sort of deal
        avg_dropoff_mape = np.mean(np.abs(dropoff_y[dropoff_mask]-dropoff_pred_y[dropoff_mask])/dropoff_y[dropoff_mask])
        avg_dropoff_rmse = np.sqrt(np.mean(np.square(dropoff_y[dropoff_mask]-dropoff_pred_y[dropoff_mask])))
    return (avg_pickup_rmse, avg_pickup_mape), (avg_dropoff_rmse, avg_dropoff_mape)

def print_time():
    print("Timestamp:", datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S"))


def caps_main(
        batch_size              = 32,
        max_epochs              = 100,
        validation_split        = 0.2,
        early_stop              = EarlyStopping(),
        model_filename          = None,
        train_dataset           = "train",
        test_dataset            = "test",
        save_filename           = None,
        initial_epoch           = 0,
        n                       = 2 ):
    model_hdf5_path = "./hdf5s/"
    sampler = file_loader.file_loader(n=n)
    modeler = models.models()
    
    window_size = n*48 # Past 2 days
    
    # Step 1. Create training dataset from raw data
    if V: print("Sampling data.")
    #gap_size = 15*12*n-1 # A week + 12 hour window, - 1 sample
    #X1, X2, y = sampler.sample_3DConv_past(datatype = train_dataset,
    X, y = sampler.sample_3DConv(datatype = train_dataset,
                                 window_size = window_size)
    #X = np.concatenate((X1, X2), axis=-1)
    
    if V: print_time()
    
    # Step 2. Compile model architecture
    #input_shape = X1[0].shape # E.g. (window_size, 10, 20, 2)
    #assert X1[0].shape == X2[0].shape ##For the new caps net
    input_shape = X[0].shape
    output_shape = y[0].shape
    if V: print("Creating model with input shape",input_shape)
    model = modeler.single_capsnet(
                input_shape = input_shape,
                routings = 3)
    if V:
        print("\nCapsule model created.")
        print_time()
    if model_filename:
        if V: print("  Loading model weights from", model_filename)
        model.load_weights(model_hdf5_path + model_filename)
        if V: print_time()
    
    # Step 3. Train model
    if V: print("Start training")
    #model.fit(x = [X1, X2], y = y,
    model.fit(x = X, y = y,
              batch_size       = batch_size,
              validation_split = validation_split,
              epochs           = max_epochs,
              initial_epoch    = initial_epoch,
              callbacks        = [early_stop])
    if V:
        print("\nModel fit complete. Starting sampling of test data for evaluation.")
        print_time()
    
    # Step 4. Test model against 'test' dataset.
    #test_X, test_y =  sampler.sample_3DConv_past(datatype = test_dataset,
    test_X, test_y =  sampler.sample_3DConv(datatype = test_dataset,
                                           window_size = window_size)
    #test_X test_y = sampler.sample_3DConv(datatype = test_dataset,
    #test_X = np.concatenate((test_X1, test_X2), axis=-1)
    if V:
        print_time()
        print("Starting evaluation.")
    
    # Step 5. Evaluation
    #y_pred = model.predict(x = [test_X1, test_X2])
    y_pred = model.predict(x = test_X)
    threshold = sampler.threshold / sampler.volume_max
    
    print("  Evaluation threshold:", sampler.threshold, " (normalized to",threshold,")")
    print("  Normalizing constant:", sampler.volume_max)
    print("  Testing on model.")
    # TODO: Continue from here!
    (prmse, pmape), (drmse, dmape) = eval_caps(test_y, y_pred, threshold)
    print("  pick-up rmse =",prmse*sampler.volume_max,", pickup mape =",pmape*100,"%")
    print("  dropoff rmse =",drmse*sampler.volume_max," dropoff mape =",dmape*100,"%")
    #MSE = model.evaluate([test_X1, test_X2], test_y)
    MSE = model.evaluate(test_X, test_y)
    print("MSE:", MSE)
    print("  (Normalized MSE:", (MSE)*sampler.volume_max**2, ")")
        
    # Step 6: Save model
    if V:
        print("\nEvaluation finished. Saving model weights.")
        print_time()
    if save_filename is None:
        currTime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        save_filename = currTime + "caps_weights.hdf5"
    model.save_weights(model_hdf5_path + save_filename)
    if V: print("Model weights saved to " + model_hdf5_path + save_filename , sep='')


def stdn_main(
        att_lstm_num            = 3, # Should be dependent on n implicitly in fileloader
        long_term_lstm_seq_len  = 3, # Should be passed dependent on n
        short_term_lstm_seq_len = 7, # Should be passed dependent on n
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
        save_filename           = None,
        initial_epoch           = 0,
        n                       = 2,
        capsule_mode            = False,
    ):
    """
    Samples raw data (from /data/*.npz files) to create training data for model,
    defines and compiles model, loads weights from model_filename if present,
    trains model with batch_size for max_epochs, then saves it to save_filename.
    """
    model_hdf5_path = "./hdf5s/"
    sampler = file_loader.file_loader(n=n)
    modeler = models.models()

    # Step 1. Create training dataset from raw data
    if V: print("Sampling data.")
    att_cnnx, att_flow, att_x, cnnx, flow, x, y = \
            sampler.sample_stdn(datatype                = train_dataset,
                                att_lstm_num            = att_lstm_num,
                                long_term_lstm_seq_len  = long_term_lstm_seq_len,
                                short_term_lstm_seq_len = short_term_lstm_seq_len,
                                last_feature_num        = n*24,
                                nbhd_size               = nbhd_size,
                                cnn_nbhd_size           = cnn_nbhd_size)
    if V: print_time()
    
    # Step 2. Compile model architecture
    if V: print("Creating model with input shape", x.shape, "/", cnnx[0].shape)
    if capsule_mode:
        model = modeler.stdcn(att_lstm_num     = att_lstm_num,           #3
                             att_lstm_seq_len = long_term_lstm_seq_len, #3
                             lstm_seq_len     = len(cnnx),              #7
                             feature_vec_len  = x.shape[-1],            #160
                             cnn_flat_size    = cnn_flat_size,          #128
                             nbhd_size        = cnnx[0].shape[1],       #7
                             nbhd_type        = cnnx[0].shape[-1],      #2
                             verbose          = V)
    else:
        model = modeler.stdn(att_lstm_num     = att_lstm_num,           #3
                             att_lstm_seq_len = long_term_lstm_seq_len, #3
                             lstm_seq_len     = len(cnnx),              #7
                             feature_vec_len  = x.shape[-1],            #160
                             cnn_flat_size    = cnn_flat_size,          #128
                             nbhd_size        = cnnx[0].shape[1],       #7
                             nbhd_type        = cnnx[0].shape[-1],      #2
                             verbose          = V)
    if V:
        print("\nSTDN model created.")
        if capsule_mode:
            print("  STDN model: STDCN (STDN with Capsule Layers)")
        print_time()
    if model_filename:
        if V: print("  Loading model weights from", model_filename)
        model.load_weights(model_hdf5_path + model_filename)
        if V: print_time()
    if V: print("Start training with input shape {1} / {0}\n".format( x.shape, cnnx[0].shape))
    
    # Step 3. Train model
    model.fit(x                = att_cnnx + att_flow + att_x + cnnx + flow + [x,],
              y                = y,
              batch_size       = batch_size,
              validation_split = validation_split,
              epochs           = max_epochs,
              callbacks        = [early_stop],
              initial_epoch    = initial_epoch
             )
    if V:
        print("\nModel fit complete. Starting sampling of test data for evaluation.")
        print_time()
    
    # Step 4. Test model against 'test' dataset.
    att_cnnx, att_flow, att_x, cnnx, flow, x, y = \
            sampler.sample_stdn(datatype                = test_dataset,
                                att_lstm_num            = att_lstm_num,
                                long_term_lstm_seq_len  = long_term_lstm_seq_len,
                                short_term_lstm_seq_len = short_term_lstm_seq_len,
                                last_feature_num        = n*24,
                                nbhd_size               = nbhd_size,
                                cnn_nbhd_size           = cnn_nbhd_size)
    if V:
        print_time()
        print("Starting evaluation.")
    
    # Step 5. Evaluation
    y_pred = model.predict(x = att_cnnx + att_flow + att_x + cnnx + flow + [x,],)
    threshold = sampler.threshold / sampler.volume_max   
    
    print("  Evaluation threshold:", sampler.threshold, " (normalized to",threshold,")")
    print("  Normalizing constant:", sampler.volume_max)
    print("  Testing on model.")
    (prmse, pmape), (drmse, dmape) = eval_lstm(y, y_pred, threshold)
    print("  pick-up rmse =",prmse*sampler.volume_max,", pickup mape =",pmape*100,"%")
    print("  dropoff rmse =",drmse*sampler.volume_max," dropoff mape =",dmape*100,"%")
    MSE = model.evaluate(att_cnnx + att_flow + att_x + cnnx + flow + [x,], y)
    print("MSE:", MSE)
    print("  (Normalized MSE:", (MSE)*sampler.volume_max**2, ")")
        
    # Step 6: Save model
    if V:
        print("\nEvaluation finished. Saving model weights.")
        print_time()
    if save_filename is None:
        currTime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        save_filename = currTime + "_weights.hdf5"
    model.save_weights(model_hdf5_path + save_filename)
    if V: print("Model weights saved to " + model_hdf5_path + save_filename , sep='')



if __name__ == "__main__":
    # Set up argument parser
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
                        help="Name of training dataset to use. (See README)",
                        type=str, nargs=1)
    parser.add_argument("--test",
                        help="Name of testing dataset to use. (See README)",
                        type=str, nargs=1)
    parser.add_argument("--save", "-s",
                        help="Location to save model weights to in hdf5s/",
                        type=str, nargs=1)
    parser.add_argument("--initialepoch",
                        help="The epoch we start training from, useful for Early Stopping callbacks. Defaults to 0.",
                        type=int, nargs=1)
    parser.add_argument("--gpunum", "-g",
                        help="GPU number. (Legacy(?))",
                        type=int, nargs=1)
    parser.add_argument("--n", "-n",
                        help="Number of samples per hour. Default 2. Don't change unless you know you need to!",
                        type=int, nargs=1)
    parser.add_argument("--arch", help="Architecture. (Default STDN)")
    # TODO; we can add a threshhold argument
    #parser.add_argument("--threshold", "-t",
    #                    help="Population threshhold. Defaults to 10",
    #                    type=int, nargs=1)
    args = parser.parse_args()
    
    # Extract arguments
    V = args.verbose
    model_filename  = None  if args.model   is None else args.model[0]
    max_epochs      = 1     if args.epochs  is None else args.epochs[0]
    batch_size      = 64    if args.batch   is None else args.batch[0]
    train_data      = "train" if args.train is None else args.train[0]
    test_data       = "test"  if args.test  is None else args.test[0]
    save_filename   = None  if args.save    is None else args.save[0]
    initial_epoch   = 0     if args.initialepoch is None else args.initialepoch[0]
    gpu_num         = None  if args.gpunum  is None else args.gpunum[0]
    n_samples_per_hour = 2  if args.n       is None else args.n[0]
    arch            = "stdn"  if args.arch  is None else args.arch.lower()
    
    if V:
        print("  Model name:",model_filename)
        print("  Max epochs:",max_epochs)
        print("  Batch size:",batch_size)
        print("  Training on",train_data,"and testing on", test_data)
        print("  Save name: ",save_filename)
        print("  Init epoch:",initial_epoch)
        print("  GPU number:",gpu_num)
        print("           n:", n_samples_per_hour)
        print("  Model arch:", arch)
    
    # Get GPU number
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

    # Start program
    
    if arch == 'stdn' or arch == 'stdcn':
        if V:
            print("\n\n####################")
            print("Starting main.py stdn_main()")
            if arch == 'stdcn':
                print("  Using Capsule variant of STDN")
            print_time()
            print("####################\n")
        stdn_main(
            att_lstm_num            = 3,
            long_term_lstm_seq_len  = max(3, 1+n_samples_per_hour),
            short_term_lstm_seq_len = floor(3.5*n_samples_per_hour),
            batch_size              = batch_size,
            max_epochs              = max_epochs,
            early_stop              = stop,
            model_filename          = model_filename,
            train_dataset           = train_data,
            test_dataset            = test_data,
            save_filename           = save_filename,
            initial_epoch           = initial_epoch,
            n                       = n_samples_per_hour,
            capsule_mode            = arch=='stdcn')
    
    elif arch == 'caps' or arch == 'capsule':
        if V:
            print("\n\n####################")
            print("Starting main.py caps_main()")
            print_time()
            print("####################\n")
        caps_main(
            batch_size      = batch_size,
            max_epochs      = max_epochs,
            early_stop      = capsstop,
            model_filename  = model_filename,
            train_dataset   = train_data,
            test_dataset    = test_data,
            save_filename   = save_filename,
            initial_epoch   = initial_epoch,
            n = n_samples_per_hour)
    else:
        print("Please define a valid architecture.")
        exit()
    
    if V:
        print("\n####################")
        print("All done!")
        print_time()
        print("####################\n\n")
