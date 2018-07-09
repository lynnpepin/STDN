# STDN


Code & Data for our Spatiotemporal Dynamic Network


## This code was tested working with the following:

  - Python 3.6
  - Ubuntu 16.04.3 LTS or RHEL 6.7
  - Keras = 2.0.5 or 2.0.8
  - tensorflow-gpu (or tensorflow) == 1.2.0 or 1.3.0 ([install guide](https://www.tensorflow.org/versions/r1.0/install/install_linux))


## Running Steps

  - Clone this repository (./STDN)
  - Create "data" folder in the same folder (STDN/data/)
  - Create "hdf5s" folder for logs (if not exist) (STDN/hdf5s/)
  - Download and extract all data files (*.npz) from data.zip and put them in "data" folder (STDN/data/*.npz)
  - Run data_subset_scripy.py to create the tiny.npz and tiny2.npz subsets for rapid experimentation (if desired.)
  - Open terminal (in STDN)
  - Run using the command line options
  - Check the output results (RMSE and MAPE). Model weightss are saved to "hdf5s" folder for further use.

## Command line:

### Arguments

* **-v verbose**: Print more information while running, such as timestamps.
* **-m model_filename**: Load a model from ./hdf5s/model_filename. If -m not specified, instantiate a new model
* **-e number-of-epochs**: Train for this many epochs, at most. Defaults to 1.
* **-b batch-size**: Train on this batch size. Defaults to 64.
* **--train train_dataset**: Train on this dataset. Defaults to 'train'.
//* *Choices are 'train', 'test', 'tiny', and 'tiny2'. (tiny is a small subset of test, tiny2 is a small subset of train.)*
* **--test test\_dataset:** Test against this dataset. Defaults to 'test'. Choices are same as train\_dataset.
* **-s save_filename**: Saves the model to ./hdf5s/save\_filename, overwriting any file that's there. Defaults to stdn(timestamp)\_weights.hdf5s.

### Examples:

Instantiate a new model, train for 1 epoch (default), and save to ./hdf5s/stdn_weights.hdf5; verbose

```
python3.6 main.py -s stdn_weights.hdf5 -v
```


Load model weights from ./hdf5s/stdn_weights.hdf5, train for 50 epochs, and save to ./hdf5s/model.hdf5; verbose

```
python3.6 main.py -m stdn_weights.hdf5 -e 50 -s model.hdf5 -v
```


Load a model from ./hdf5s/model.hdf5, train for 10 epochs on the "tiny" dataset with a batch size of 128, test on the "tiny2" dataset, and save to "./hdf5s/model.hdf5"

```
python3.6 main.py -m model.hdf5 -e 10 --train tiny -b 128 -test tiny2 -s model.hdf5
```
