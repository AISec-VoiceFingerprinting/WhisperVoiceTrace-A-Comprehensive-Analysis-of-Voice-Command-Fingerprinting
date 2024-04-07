#!/usr/bin/env python3
import os
import numpy as np
from sys import exit
from os.path import dirname, isdir, exists
from os import walk, makedirs
from scapy.layers.inet import IP
from scapy.utils import PcapReader
from argparse import ArgumentParser
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import concatenate
from tensorflow.keras.layers import Input, Concatenate, Dense, BatchNormalization
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import to_categorical
from pickle import load, dump
from df import DFNet
from multiprocessing import Pool
import math


"""
Description:    Train the model using the VAV1 feature.
Usage:          python3 run_VAV1.py --load-VAV1 ./VAV1.pkl --load-classes ./y.pkl
"""

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= ""



def check_args(args):
    if args.load_VAV1 is None and args.load_classes is None:
        exit('Please run the program using --load-VAV1, and --load-classes')


def build_deep_fingerprinting_model(INPUT_SHAPE, NUM_CLASSES):
    model = DFNet.build(INPUT_SHAPE, NUM_CLASSES)
    adam = Adam(lr=0.005)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    return model




def train_WhiVo_Model(VAV1_2d, INPUT_SHAPE, NUM_CLASSES, VAV1_weights, y):
    
    if not isdir("./VAV1_weights"):
        makedirs("./VAV1_weights")
    if not isdir("./pics"):
        makedirs("./pics")

    early_stopping = EarlyStopping(monitor="val_accuracy", patience=100, restore_best_weights=True)

    checkpoint1 = ModelCheckpoint('./VAV1_weights/VAV1-model-{epoch:03d}-{accuracy:03f}-{val_accuracy:03f}.h5',
                                verbose=1, monitor='val_accuracy',
                                save_best_only=True, mode='auto')
    tensorboard_callback = TensorBoard(log_dir="./logs")

    # Create DF model
    model = build_deep_fingerprinting_model(INPUT_SHAPE, NUM_CLASSES)

    te_cut = int(len(y)*.9)

    #  If pre-trained weights are available, pass; otherwise, train
    batch_size = 256
    if VAV1_weights is None:
        df_size_history = model.fit(VAV1_2d[:te_cut], to_categorical(y[:te_cut]),
                                    batch_size=batch_size,
                                    epochs=600,
                                    validation_split=0.10,
                                    verbose=False,
                                    callbacks=[checkpoint1, early_stopping])
    else:
        model.load_weights(VAV1_weights)


    #  Make sure to not train either model any further
    model.trainable = False

    # Save the model architecture
    from keras.utils.vis_utils import plot_model
    plot_model(model, to_file='./pics/VAV1_model.png', show_shapes='True')




def parse_arguments():
    """parse command-line arguments"""
    parser = ArgumentParser()
    parser.add_argument("--load-VAV1", metavar=' ',
                        help='filepath containing VAV1 arrays')
    parser.add_argument("--load-classes", metavar=' ', help='filepath containing training (classes)')
    parser.add_argument("--load-weights-VAV1", metavar=' ', help='filepath to VAV1 weights to load into model', default=None)
    args = parser.parse_args()
    check_args(args)
    return args

if __name__ == '__main__':
   
    args = parse_arguments()
    NUM_CLASSES = None


    if args.load_VAV1 is None:
        exit('Please provide values for arguments (--load-VAV1) when loading in your own data')
    else:
        # Load VAV1 feature and label
        with open(args.load_VAV1, "rb") as fp:
            VAV1_2d = load(fp)
        with open(args.load_classes, "rb") as fp:
            y = load(fp)
            NUM_CLASSES = len(np.unique(y))

    INPUT_SHAPE = (7500, 1)


    # Train WhiVo model
    VAV1_weights = args.load_weights_VAV1
    train_WhiVo_Model(VAV1_2d, INPUT_SHAPE, NUM_CLASSES, VAV1_weights, y)
