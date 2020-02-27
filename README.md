# iASL-Backend

This repository is intended to contain the back-end material for the iASL application. Specifically, it will contain the machine learning model for the neural network used in this application.

The main code for running the training portion of the model can be found in the `scripts` folder. The main script, `train.sh`, runs the training network and saves the model to the folder `output/p1_train`. The training process results in three files being saved, as well as one additional weights file for every epoch.

# Training

Included in the data_processing directory is a `collect_data.py` file. This contains a class responsible for batch data generation. This class takes in class mapping which is a dictionary where the key is the label and the value is the integer index value. This file creates a master mapping of each image file and its label, and allows for parallel batch processing during training. This data generator is necessary for the training script. To train a model on a dataset, a file list with a new line seperated list of file names should be provided. Along with this, the corresponding ref file should be provided which contains the label in the same order as the image. Use the `gen_labels.py` script to generate this after the file list has been created. Some hyper parameter values can be found in the parameter file in `params/params_train.txt`. Each epoch, the model will output to `output/p1_train` a weights file. To run the training script, simply edit the parameter file as necessary and run `train.sh`. This will source the `_runtime_env.sh` which contains environment variables necessary for the script to run. Note that `train.py` will fail to run if the runtime file is not sourced.

# Realtime Detection

The `realtime-detection.py` script only takes a parameter file. In this file should be a model directory (relative to the iASL_OUT directory) where the model and weights are stored. This script will load these files to use as the model. On start-up, there will be a green box in the top left of the screen. Place your hand there within the first three seconds. This will activate the object tracking so the region of interest will follow your hand. Each frame displayed will be sent to the model for classsification. On detection, the confidence and label will be displayed in a black box. Running `detect.sh` will provide the parameter file and source the runtime file, so it is suggested to run the bash file as opposed to running the python script directly.
