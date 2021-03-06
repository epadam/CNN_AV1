import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import optimizers, backend
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten, Concatenate, Conv2D, MaxPooling2D, Lambda, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
#from tensorflow.keras.utils import plot_model

num_classes = 9
block_size = 16
NUM_CHANNELS = 1
NUM_EXT_FEATURES = 1


def sub_mean(x):
    x = x/255
    x = x - backend.mean(x)   
    return x


def net():

    data = Input(shape=(block_size,block_size,NUM_CHANNELS))

    data_norm = Lambda(sub_mean)(data)

    conv1 = Conv2D(32, (3, 3), strides =(1,1),padding='valid', activation='relu')(data_norm)
    conv1_dropout = Dropout(rate=0.5)(conv1)
 
    conv2 = Conv2D(64, (3, 3), strides =(1,1), activation='relu', padding='valid')(conv1_dropout)
    conv2_dropout = Dropout(rate=0.5)(conv2)

    pooling1 = MaxPooling2D(pool_size=(2, 2))(conv2_dropout)
    
    flat2 = Flatten()(pooling1)

    qp = Input(shape=(1,))
    qp_n = Lambda(lambda x: x/255)(qp)

    concat = Concatenate(axis=1)([flat2, qp_n])

    fc1 = Dense(128, activation='relu')(concat)
    fc1_d = Dropout(rate=0.5)(fc1)
    
    fc1_qp = Concatenate(axis=1)([fc1_d, qp_n])

    #fc2 = Dense(48, activation='relu')(fc1_qp)
    #fc2_qp = Concatenate(axis=1)([fc2, qp_n])

    output = Dense(num_classes, activation='softmax')(fc1_qp)

    model = Model(inputs=[data,qp], outputs=output)

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=['accuracy'])
    
    #plot_model(model, to_file='model.png')
    
    return model

