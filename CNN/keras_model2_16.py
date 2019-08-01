import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import optimizers, backend
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten, Concatenate, Conv2D, MaxPooling2D, Lambda, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard

num_classes = 10
block_size = 16
NUM_CHANNELS = 1
NUM_EXT_FEATURES = 1


def sub_mean(x):
    x = x/255
    x = x - backend.mean(x)   
    return x


def net():

    data = Input(shape=(block_size,block_size,NUM_CHANNELS))

    qp = Input(shape=(1,))

    qp_n = Lambda(lambda x: x/255)(qp)

    data_norm = Lambda(sub_mean)(data)

    conv1 = Conv2D(16, (4, 4), strides =(4,4),padding='valid', activation='relu')(data_norm)
    conv1_Norm = BatchNormalization()(conv1)

    conv2 = Conv2D(24, (2, 2), strides =(2,2), activation='relu', padding='valid')(conv1_Norm)
    conv2_Norm = BatchNormalization()(conv2)
    #flat2 = Flatten()(conv2_Norm)

    conv3 = Conv2D(32, (2, 2), strides =(2,2), activation='relu', padding='valid')(conv2_Norm)
    conv3_Norm = BatchNormalization()(conv3)
    flat3 = Flatten()(conv3_Norm)

    #concat = Concatenate(axis=1)([flat2, flat3])
    flat3_qp = Concatenate(axis=1)([flat3, qp_n])
    fc1 = Dense(16, activation='relu')(flat3_qp)
    fc1_qp = Concatenate(axis=1)([fc1, qp_n])

    #fc2 = Dense(48, activation='relu')(fc1_qp)
    #fc2_qp = Concatenate(axis=1)([fc2, qp_n])
    #model.add(Dropout(rate=0.5))

    output = Dense(num_classes, activation='softmax')(fc1_qp)

    model = Model(inputs=[data,qp], outputs=output)

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=['accuracy'])
    
    return model

