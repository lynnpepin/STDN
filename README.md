# STDN


Code & Data for the Spatiotemporal Dynamic Network described in [H. Yao, X. Tang, et al 2018 "Modeling Spatial-Temporal Dynamics for Traffic Prediction"](https://arxiv.org/abs/1803.01254)


**This code was tested working with the following:**

  - Python 3.6
  - Ubuntu 16.04.3 LTS or RHEL 6.7
  - Keras = 2.0.5 or 2.0.8
  - tensorflow-gpu (or tensorflow) == 1.2.0 or 1.3.0 ([install guide](https://www.tensorflow.org/versions/r1.0/install/install_linux))

Python >= 3.5, Keras >= 2.0.5, and tensorflow >= 1.0.0 should work, but have not been tested.

## Set up steps (Linux)

  - Clone this repository (./STDN)
  - Open STDN directory in terminal
  - Make init.sh executable (chmod +x init.sh)
  - Run init.sh (./init.sh)

## Running via Command line

  - Run using the command line options
  - Check the output results (RMSE and MAPE).
  - Model weightss are saved to "hdf5s" folder for further use.

### Arguments

* **-v verbose**: Print more information while running, such as timestamps.
* **-m model_filename**: Load a model from ./hdf5s/model_filename. If -m not specified, instantiate a new model
* **-e number-of-epochs**: Train for this many epochs, at most. Defaults to 1.
* **-b batch-size**: Train on this batch size. Defaults to 64.
* **--train train_dataset**: Train on this dataset. Defaults to 'train'.
//* *Choices are 'train', 'test', 'tiny', and 'tiny2'. (tiny is a small subset of test, tiny2 is a small subset of train.)*
//* You can choose to use the Manhattan dataset, as man\_train/test/tiny/tiny2\_1\2\4. (E.g. man\_train\_2 is the manhattan training set with 2 timeslots per hour.)
* **--test test\_dataset:** Test against this dataset. Defaults to 'test'. Choices are same as train\_dataset.
* **-s save_filename**: Saves the model to ./hdf5s/save\_filename, overwriting any file that's there. Defaults to stdn(timestamp)\_weights.hdf5s.
* **--initialepoch:** The epoch we start training from, useful for Early Stopping callbacks. Defaults to 0.
* **-n:** Number of samples per hour; don't change unless you know what you're doing!

### Examples:

Instantiate a new model, train for 1 epoch (default), and save to ./hdf5s/stdn_weights.hdf5; verbose

```
python3.6 main.py -s stdn_weights.hdf5 -v
```


Load model weights from ./hdf5s/stdn_weights.hdf5, train for 50 epochs, and save to ./hdf5s/model.hdf5; verbose

```
python3.6 main.py -m stdn_weights.hdf5 -e 50 -s model.hdf5 -v
```


Load a model from ./hdf5s/model.hdf5, start from epoch 100, train for 10 epochs on the "tiny" dataset with a batch size of 128, test on the "tiny2" dataset, and save to "./hdf5s/model.hdf5".

```
python3.6 main.py -m model.hdf5 --initialepoch 100 -e 10 --train tiny -b 128 -test tiny2 -s model.hdf5
```
