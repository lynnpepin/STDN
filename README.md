# STDN


Code & Data for our Spatiotemporal Dynamic Network



Requirements

  - Python 3.6 (Recommend Anaconda)
  - Ubuntu 16.04.3 LTS
  - Keras >= 2.0.8
  - tensorflow-gpu (or tensorflow) == 1.3.0 ([install guide](https://www.tensorflow.org/versions/r1.0/install/install_linux))


Running Steps
  - Download all codes (*.py) and put them in the same folder (let's name it "stdn") (stdn/*.py)
  - Create "data" folder in the same folder (stdn/data/)
  - Create "hdf5s" folder for logs (if not exist) (stdn/hdf5s/)
  - Download and extract all data files (*.npz) from data.zip and put them in "data" folder (stdn/data/*.npz)
  - *New step*: Run data_subset_scripy.py to create the tiny.npz and tiny2.npz subsets for rapid experimentation (if desired.)
  - Open terminal in the same folder (stdn/)
  - *NOTE*: Some things are coded into main_tweak.py, which is now used instead of main.py.
  - *These include DEBUG flags, VERBOSE flags, and which datasets to use for training and testing. main.py should not be used.*
  - Run with "python main.py --stdn" for NYC taxi dataset, or "python main_bike.py --stdn" for NYC bike dataset
  - Check the output results (RMSE and MAPE). Models are saved to "hdf5s" folder for further use.
