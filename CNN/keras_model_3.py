import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import optimizers, backend
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten, Concatenate, Conv2D, MaxPooling2D, Lambda
from tensorflow.keras.callbacks import TensorBoard

batch_size = 1
num_classes = 10
block_size = 16
NUM_CHANNELS = 1
epochs = 1
sample_file = 'sample10.txt'
label_file = 'label10.txt'
qp_file = 'qp10.txt'

#read samples
with open(sample_file, 'rb') as f:
    pixels = f.read()
    raw = np.frombuffer(pixels, dtype = np.float)
    raw = np.reshape(raw, [-1, block_size, block_size, NUM_CHANNELS])
    #print(raw)
    print(np.shape(raw))
    raw2=raw/255
    rmean = np.mean(raw2[1])
    print(rmean)
    raw3=raw2 - rmean
    #print(raw3)  
   
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
    print(qps/255)
    print(np.shape(qps)) 



data = Input(shape=(block_size,block_size,NUM_CHANNELS))

qp = Input(shape=(1,))

qp_n = Lambda(lambda x: x/255)(qp)

qp_n_d = Lambda(lambda x: tf.Print(x, [x], message='qp=', summarize=-1))(qp_n)

data_display = Lambda(lambda x: tf.Print(x, [x], message='data=', summarize=-1))(data)


def sub_mean(x):
    x = x/255
    x = x - backend.mean(x)   
    return x


data_norm = Lambda(sub_mean)(data_display)

data_n_display = Lambda(lambda x: tf.Print(x, [x], message='data_norm=', summarize=-1))(data_norm)


conv1 = Conv2D(16, (4, 4), strides =(4,4),padding='valid', activation='relu')(data_n_display)

conv2 = Conv2D(24, (2, 2), strides =(2,2), activation='relu', padding='valid')(conv1)
flat2 = Flatten()(conv2)

conv3 = Conv2D(32, (2, 2), strides =(2,2), activation='relu', padding='valid')(conv2)
flat3 = Flatten()(conv3)

concat = Concatenate(axis=1)([flat2, flat3])

fc1 = Dense(64, activation='relu')(concat)
fc1_qp = Concatenate(axis=1)([fc1, qp_n_d])

fc2 = Dense(48, activation='relu')(fc1_qp)
fc2_qp = Concatenate(axis=1)([fc2, qp_n_d])
#model.add(Dropout(rate=0.5))

output = Dense(num_classes, activation='softmax')(fc2_qp)



model = Model(inputs=[data,qp], outputs=output)

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])


tbCallBack = TensorBoard(log_dir='./logs',  # log ç›®å½•
                 histogram_freq=0,  # æŒ‰ç…§ä½•ç­‰é¢‘çŽ‡ï¼ˆepochï¼‰æ¥è®¡ç®—ç›´æ–¹å›¾ï¼Œ0ä¸ºä¸è®¡ç®—
#                  batch_size=32,     # ç”¨å¤šå¤§é‡çš„æ•°æ®è®¡ç®—ç›´æ–¹å›¾
                 write_graph=True,  # æ˜¯å¦å­˜å‚¨ç½‘ç»œç»“æž„å›¾
                 write_grads=True, # æ˜¯å¦å¯è§†åŒ–æ¢¯åº¦ç›´æ–¹å›¾
                 write_images=True,# æ˜¯å¦å¯è§†åŒ–å‚æ•°
                 embeddings_freq=0, 
                 embeddings_layer_names=None, 
                 embeddings_metadata=None)


history = model.fit([raw, qps], single_label, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[tbCallBack])

model.save('my_model.h5') 



'''

tsample_file = 'Library_frames0_ec_samples_intra_16_intra.txt'
tlabel_file = 'Library_frames0_ec_labels_16_intra.txt'
tqp_file = 'Library_frames0_ec_qps_16_intra.txt'

#read samples
with open(tsample_file, 'rb') as f:
    pixels = f.read()
    raw = np.frombuffer(pixels, dtype = np.float)
    traw = np.reshape(raw, [-1, block_size, block_size, NUM_CHANNELS])
    print(np.shape(traw))  
   
#read labels
with open(tlabel_file, 'r') as f_single_label:
    single_label = f_single_label.read()    
    single_label =np.fromstring (single_label, dtype=np.int32 ,sep=' ')
    single_label = np.reshape(single_label, [-1])
    tsingle_label = keras.utils.to_categorical(single_label, num_classes)
    print(np.shape(tsingle_label))

#labels_10 = np.loadtxt("labels_10_64.txt", dtype=float)

#read qps
with open(tqp_file, 'r') as f_qp:
    qps = f_qp.read()
    qps =np.fromstring (qps, dtype=np.float, sep=' ')
    tqps = np.reshape(qps, [-1,1]) 
    print(np.shape(tqps)) 

# loss_and_metrics = model.evaluate([traw, tqps], tsingle_label, batch_size=32)
    
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



classes = model.predict([traw, tqps], batch_size=32)

print(classes)

'''

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
