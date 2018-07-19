#!/bin/bash
mkdir data
mkdir hdf5s
unzip data.zip -d data
unzip man_data.zip -d data
python3 data_subset_script.py
