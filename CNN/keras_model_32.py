import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import optimizers, backend
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten, Concatenate, Conv2D, AveragePooling2D, Lambda
from tensorflow.keras.callbacks import TensorBoard

num_classes = 9
block_size = 32
NUM_CHANNELS = 1
NUM_EXT_FEATURES = 1


def sub_mean(x):
    x = x/255
    x = x - backend.mean(x)   
    return x


def net():

    data = Input(shape=(block_size,block_size,NUM_CHANNELS))

    data_norm = Lambda(sub_mean)(data)

    data_pooling = AveragePooling2D(pool_size=(2, 2),padding='valid')(data_norm)   

    conv1 = Conv2D(16, (4, 4), strides =(4,4),padding='valid', activation='relu')(data_pooling)
    conv1_d = Dropout(0.7)(conv1)

    conv2 = Conv2D(24, (2, 2), strides =(2,2), activation='relu', padding='valid')(conv1_d)
    conv2_d = Dropout(0.7)(conv2)
    flat2 = Flatten()(conv2_d)

    conv3 = Conv2D(32, (2, 2), strides =(2,2), activation='relu', padding='valid')(conv2_d)
    conv3_d = Dropout(0.7)(conv3)
    flat3 = Flatten()(conv3_d)

    qp = Input(shape=(1,))
    qp_n = Lambda(lambda x: x/255)(qp)

    concat = Concatenate(axis=1)([flat2, flat3, qp_n])

    fc1 = Dense(64, activation='relu')(concat)
    fc1_d = Dropout(0.7)(fc1)
    fc1_qp = Concatenate(axis=1)([fc1_d, qp_n])

    fc2 = Dense(48, activation='relu')(fc1_qp)
    fc2_d = Dropout(0.7)(fc2)
    fc2_qp = Concatenate(axis=1)([fc2_d, qp_n])

    output = Dense(num_classes, activation='softmax')(fc2_qp)

    model = Model(inputs=[data,qp], outputs=output)

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=['accuracy'])
    
    return model

