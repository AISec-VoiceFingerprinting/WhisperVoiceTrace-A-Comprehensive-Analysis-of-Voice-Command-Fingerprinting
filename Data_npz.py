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
Description:    Convert the time, size*direction features stored in the npz file into VAV1, VAV2, and Size features and save them as a pickle file.
Usage:          python3 Data_npz.py --input-data ./data_mon_300.npz
"""


def feature_extraction(file_path, MAX_SIZE):
    
    data = np.load(file_path, allow_pickle=True)
    data_time = data['time']
    data_size_dir = data['size_direction']
    data_y = data['y']

    VAV1_2d = []
    VAV2_2d = []
    Size_2d = []

    for i in range(len(data_time)):

        # Convert to VAV1, VAV2
        VAV1 = VAV1_f(data_time[i], data_size_dir[i])
        VAV2 = VAV2_f(data_time[i], data_size_dir[i])
        Size = list(data_size_dir[i])

        # Padding
        Size.extend([0]*(MAX_SIZE-len(Size)))
        if len(Size) > MAX_SIZE:
            Size = Size[:MAX_SIZE]

        VAV1 = np.asarray(VAV1)
        VAV2 = np.asarray(VAV2)
        Size = np.asarray(Size)
        VAV1_2d.append(VAV1)
        VAV2_2d.append(VAV2)
        Size_2d.append(Size)

    VAV1_2d = np.array(VAV1_2d)
    VAV2_2d = np.array(VAV2_2d)
    Size_2d = np.array(Size_2d)

    VAV1_2d, VAV2_2d, Size_2d, data_y = shuffle(VAV1_2d, VAV2_2d, Size_2d, np.array(data_y), random_state=100)

    return VAV1_2d, VAV2_2d, Size_2d, data_y, len(np.unique(data_y))



MaxTime = 60000
MaxSize = 7500


def VAV2_f(times, sizes):
    feature1 = [0 for _ in range(MaxSize)]
    feature2 = [0 for _ in range(MaxSize)]
    for i in range(0, len(times)):
        if  sizes[i] > 0:
            if times[i] >= MaxTime:
                feature1[-1] += sizes[i]
            else:
                idx = int(times[i] * (MaxSize - 1) / MaxTime)
                feature1[idx] += sizes[i]
        if sizes[i] < 0:
            if times[i] >= MaxTime:
                feature2[-1] += sizes[i]
            else:
                idx = int(times[i] * (MaxSize - 1) / MaxTime)
                feature2[idx] += sizes[i]

    return feature1 + feature2

def VAV1_f(times, sizes):
    feature1 = [0 for _ in range(MaxSize)]
    for i in range(0, len(times)):
        if times[i] >= MaxTime:
            feature1[-1] += sizes[i]
        else:
            idx = int(times[i] * (MaxSize - 1) / MaxTime)
            feature1[idx] += sizes[i]

    return feature1



def check_args(args):
    if args.input_data is None and args.load_size is None and args.load_classes is None:
        exit('Please run the program using either --input-data, --load-size, and --load-classes')



def parse_arguments():
    """parse command-line arguments"""
    parser = ArgumentParser()
    parser.add_argument("--input-data", metavar=' ', help='filepath of folders containing arrays')
    parser.add_argument("--load-VAV1", metavar=' ',
                        help='filepath containing VAV1_arrays')
    parser.add_argument("--load-VAV2", metavar=' ',
                        help='filepath containing VAV2_arrays')
    parser.add_argument("--load-Size", metavar=' ',
                        help='filepath containing Size_arrays')
    parser.add_argument("--load-classes", metavar=' ', help='filepath containing training classes')
    parser.add_argument("--load-weights-VAV1", metavar=' ', help='filepath to VAV1 weights to load into model', default=None)
    parser.add_argument("--load-weights-VAV2", metavar=' ', help='filepath to VAV2 weights to load into model', default=None)
    parser.add_argument("--load-weights-Size", metavar=' ', help='filepath to Size weights to load into model', default=None)
    args = parser.parse_args()
    check_args(args)
    return args

if __name__ == '__main__':
   
    args = parse_arguments()
    MAX_SIZE = 2000  # Default
    NUM_CLASSES = None

    if args.input_data is not None:

        # Extract features
        VAV1_2d, VAV2_2d, Size_2d, y, NUM_CLASSES = feature_extraction(args.input_data, MAX_SIZE)

        VAV1_2d = VAV1_2d[..., np.newaxis]
        VAV2_2d = VAV2_2d[..., np.newaxis]
        Size_2d = Size_2d[..., np.newaxis]
        
        # Save the data as pickle files
        with open("VAV1.pkl", "wb") as fp:
            dump(VAV1_2d, fp)
        with open("VAV2.pkl", "wb") as fp:
            dump(VAV2_2d, fp)
        with open("Size.pkl", "wb") as fp:
            dump(Size_2d, fp)
        with open("y.pkl", "wb") as fp:
            dump(y, fp)



