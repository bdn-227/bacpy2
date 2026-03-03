#!/bin/bash

# create the environment and install dependencies
conda env create -f bacpy_new.yml

# activate the environment
conda activate bacpy_new

# install bacpy package
pip install -e .
