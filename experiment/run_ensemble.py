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
from keras.utils.vis_utils import plot_model


"""
Description:    Train and test an ensemble model composed of Dense layers using the outputs of three individual models
Usage:          python3 run.py --load-VAV2 ./VAV2.pkl --load-Size ./Size.pkl --load-VAV1 ./VAV1.pkl --load-classes ./y.pkl
                               --load-weights-VAV2 ./VAV2_weights/ooooooo.h5
                               --load-weights-Size ./Size_weights/ooooooo.h5
                               --load-weights-VAV1 ./VAV1_weights/ooooooo.h5
"""

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "4"


def check_args(args):
    if args.load_VAV1 is None and args.load_classes is None:
        exit('Please run the program using either --input-data or --load-VAV1, --load-VAV2, and --load-classes')


def build_deep_fingerprinting_model(INPUT_SHAPE, NUM_CLASSES):
    model = DFNet.build(INPUT_SHAPE, NUM_CLASSES)
    adam = Adam(lr=0.005)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    return model


def build_ensemble_model(shape, NUM_CLASSES):
    """
    This ensemble model is based on the Deep Fingerprinting Model's flatten and dense layers
    before classification.
    :param shape: The shape of the ensembled training data
    :return: The ensemble model
    """
    initializer = tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='untruncated_normal')
    model = Sequential()
    model.add(Input(shape=shape))
    model.add(Dense(4072, kernel_initializer=glorot_uniform(seed=0), name='fc1'))
    model.add(Activation('relu', name='fc1_act'))
    model.add(Dropout(0.7, name='fc1_dropout'))
    model.add(Dense(4072, kernel_initializer=glorot_uniform(seed=0), name='fc2'))
    model.add(Activation('relu', name='fc2_act'))
    model.add(Dropout(0.7, name='fc2_dropout'))
    model.add(Dense(NUM_CLASSES, kernel_initializer=glorot_uniform(seed=0), name='fc3'))
    model.add(Activation('softmax', name="softmax"))

    sgd = SGD(lr=0.003, nesterov=True, momentum=0.5)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model


def train_WhiVo_Model(VAV2_2d, Size_2d, VAV1_2d, NUM_CLASSES, VAV2_weights, Size_weights, VAV1_weights, ensemble_weights, y):

    if not isdir("./ensemble_weights"):
        makedirs("./ensemble_weights")
    if not isdir("./pics"):
        makedirs("./pics")

    early_stopping = EarlyStopping(monitor="val_accuracy", patience=100, restore_best_weights=True)
    checkpoint4 = ModelCheckpoint('./ensemble_weights/ensemble-model-{epoch:03d}-{accuracy:03f}-{val_accuracy:03f}.h5',
                                verbose=1, monitor='val_accuracy',
                                save_best_only=True, mode='auto')
    tensorboard_callback = TensorBoard(log_dir="./logs")

    # Create Three DF models
    model1 = build_deep_fingerprinting_model((15000,1), NUM_CLASSES)
    model2 = build_deep_fingerprinting_model((2000,1), NUM_CLASSES)
    model3 = build_deep_fingerprinting_model((7500,1), NUM_CLASSES)

    te_cut = int(len(y)*.9)

    #  Load pre-trained weights
    model1.load_weights(VAV2_weights)
    model2.load_weights(Size_weights)
    model3.load_weights(VAV1_weights)

    #  Make sure to not train either model any further
    model1.trainable = False
    model2.trainable = False
    model3.trainable = False


    print("Getting Flatten layer using the size array")
    #  Create a new model that takes in (MAX_SIZE, 1) and outputs the flatten layers for size
    flatten_model1 = Model(inputs=model1.input, outputs=model1.get_layer('flatten').output)
    outputs1 = flatten_model1.predict(VAV2_2d, verbose=1)  # (N, 1024)

    print("Getting Flatten layer using the size array")
    #  Create a new model that takes in (MAX_SIZE, 1) and outputs the flatten layers for size
    flatten_model2 = Model(inputs=model2.input, outputs=model2.get_layer('flatten').output)
    outputs2 = flatten_model2.predict(Size_2d, verbose=1)  # (N, 1024)

    print("Getting Flatten layer using the size array")
    #  Create a new model that takes in (MAX_SIZE, 1) and outputs the flatten layers for size
    flatten_model3 = Model(inputs=model3.input, outputs=model3.get_layer('flatten').output)
    outputs3 = flatten_model3.predict(VAV1_2d, verbose=1)  # (N, 1024)


    ensemble_input = np.concatenate((outputs1, outputs2, outputs3), axis=1)
    model4 = build_ensemble_model((ensemble_input.shape[1],), NUM_CLASSES)
    model4.summary()

    #  If pre-trained weights are available, pass; otherwise, train
    if ensemble_weights is None:
        model4.fit(x=ensemble_input[:te_cut], y=to_categorical(y[:te_cut]),
                batch_size=256,
                epochs=600,
                validation_split=0.10,
                verbose=False,
                callbacks=[checkpoint4, early_stopping])
    else:
        model4.load_weights(ensemble_weights)

    # Save the model architecture
    plot_model(model4, to_file='./pics/ensemble_model.png', show_shapes='True')

    # Testing the models
    result_size_Ours = model1.evaluate(x=VAV2_2d[te_cut:], y=to_categorical(y[te_cut:]), verbose=0)
    result_size_Shame = model2.evaluate(x=Size_2d[te_cut:], y=to_categorical(y[te_cut:]), verbose=0)
    result_size_Ours_sum = model3.evaluate(x=VAV1_2d[te_cut:], y=to_categorical(y[te_cut:]), verbose=0)
    result_ensemble = model4.evaluate(x=ensemble_input[te_cut:], y=to_categorical(y[te_cut:]), verbose=0)
  
    print(result_size_Ours)
    print(result_size_Shame)
    print(result_size_Ours_sum)
    print(result_ensemble)



def parse_arguments():
    """parse command-line arguments"""
    parser = ArgumentParser()
    parser.add_argument("--input-data", metavar=' ', help='filepath of folders containing arrays')
    parser.add_argument("--load-VAV2", metavar=' ',
                        help='filepath containing VAV2 arrays')
    parser.add_argument("--load-Size", metavar=' ',
                        help='filepath containing Size arrays')
    parser.add_argument("--load-VAV1", metavar=' ',
                        help='filepath containing VAV1 arrays')
    parser.add_argument("--load-classes", metavar=' ', help='filepath containing training (classes)')
    parser.add_argument("--load-weights-VAV2", metavar=' ', help='filepath to VAV2 weights to load into model')
    parser.add_argument("--load-weights-Size", metavar=' ', help='filepath to Size weights to load into model')
    parser.add_argument("--load-weights-VAV1", metavar=' ', help='filepath to VAV1 weights to load into model')
    parser.add_argument("--load-weights-ensemble", metavar=' ', help='filepath to ensemble weights to load into model', default=None)
    args = parser.parse_args()
    check_args(args)
    return args

if __name__ == '__main__':
   
    args = parse_arguments()
    NUM_CLASSES = None

    # Load Features and label
    with open(args.load_VAV2, "rb") as fp:
        VAV2_2d = load(fp)
    with open(args.load_Size, "rb") as fp:
        Size_2d = load(fp)
    with open(args.load_VAV1, "rb") as fp:
        VAV1_2d= load(fp)
    with open(args.load_classes, "rb") as fp:
        y = load(fp)
        print(len(y))
        NUM_CLASSES = len(np.unique(y))

    # Load pre-trained weights and train
    VAV2_weights = args.load_weights_VAV2
    Size_weights = args.load_weights_Size
    VAV1_weights = args.load_weights_VAV1
    ensemble_weights = args.load_weights_ensemble
    train_WhiVo_Model(VAV2_2d, Size_2d, VAV1_2d, NUM_CLASSES, VAV2_weights, Size_weights, VAV1_weights, ensemble_weights,y)
