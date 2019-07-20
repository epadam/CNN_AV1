import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten, Concatenate, Conv2D, MaxPooling2D
from tensorflow.keras import optimizers

batch_size = 32
num_classes = 10
block_size = 16
NUM_CHANNELS = 1
epochs = 2
sample_file = 'Bund_Nightscape_frames0_ec_samples_intra_16_intra.txt'
label_file = 'Bund_Nightscape_frames0_ec_labels_16_intra.txt'
qp_file = 'Bund_Nightscape_frames0_ec_qps_16_intra.txt'

#read samples
with open(sample_file, 'rb') as f:
    pixels = f.read()
    raw = np.frombuffer(pixels, dtype = np.float)
    raw = np.reshape(raw, [-1, block_size, block_size, NUM_CHANNELS])
    raw = raw/255
    print(np.shape(raw))  
   
#read labels
with open(label_file, 'r') as f_single_label:
    single_label = f_single_label.read()    
    single_label =np.fromstring (single_label, dtype=np.int32 ,sep=' ')
    single_label = np.reshape(single_label, [-1])
    single_label = keras.utils.to_categorical(single_label, num_classes)
    print(np.shape(single_label))

#labels_10 = np.loadtxt("labels_10_64.txt", dtype=float)

#read qps
with open(qp_file, 'r') as f_qp:
    qps = f_qp.read()
    qps =np.fromstring (qps, dtype=np.float, sep=' ')
    qps = np.reshape(qps, [-1,1]) 
    print(np.shape(qps)) 

data = Input(shape=(block_size,block_size,NUM_CHANNELS))
qp = Input(shape=(1,))

conv1 = Conv2D(16, (4, 4), strides =(4,4),padding='valid', activation='relu')(data)
conv2 = Conv2D(24, (2, 2), strides =(2,2), activation='relu', padding='valid')(conv1)
flat2 = Flatten()(conv2)


conv3 = Conv2D(32, (2, 2), strides =(2,2), activation='relu', padding='valid')(conv2)
flat3 = Flatten()(conv3)


concat = Concatenate(axis=1)([flat2, flat3])

fc1 = Dense(64, activation='relu')(concat)
fc1_qp = Concatenate(axis=1)([fc1, qp])

fc2 = Dense(48, activation='relu')(fc1_qp)
fc2_qp = Concatenate(axis=1)([fc2, qp])
# #model.add(Dropout(rate=0.5))
output = Dense(num_classes, activation='softmax')(fc2_qp)
# #model.add(Activation('softmax'))

model = Model(inputs=[data,qp], outputs=output)

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])


model.fit([raw, qps], single_label, batch_size=batch_size, epochs=epochs,
           verbose=1)

#loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)
#classes = model.predict(x_test, batch_size=32)


# qp = Input(shape=(1,))
# model = Sequential()

# conv1 = Conv2D(16, (4, 4), strides =(4,4),padding='valid', activation='relu', input_shape = (block_size, block_size, NUM_CHANNELS))
# model.add(conv1)

# conv2 = Conv2D(24, (2, 2), strides =(2,2), activation='relu', padding='valid')
# model.add(conv2)
# flat2 = Flatten()(conv2)

# conv3 = Conv2D(32, (2, 2), strides =(2,2), activation='relu', padding='valid')
# model.add(conv3)
# flat3 = Flatten()(conv3)


# model.add(Concatenate([flat2, flat3], axis=1))

# #model.add(Concatenate([x, qp]))
# model.add(Dense(16, activation='relu'))
# #model.add(Dropout(rate=0.5))
# model.add(Dense(num_classes), activation='softmax') 
# #model.add(Activation('softmax'))

# model.summary()

# optimizers.Adam(lr=0.1)

# model.compile(loss='categorical_crossentropy',
#               optimizer='Adam',
#               metrics=['accuracy'])


# model.fit(data, single_label, batch_size=batch_size, epochs=epochs,
#            verbose=1)