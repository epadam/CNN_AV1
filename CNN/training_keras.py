import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard
import keras_model2_16 as cnn16
import keras_model_32 as cnn32
import keras_model_64 as cnn64

batch_size = 128
num_classes = 10
block_size = 16
NUM_CHANNELS = 1
epochs = 300
sample_file = '1080_samples_16_intra.txt'
label_file = '1080_labels_16_intra.txt'
qp_file = '1080_qps_16_intra.txt'


#read samples
with open(sample_file, 'rb') as f:
    pixels = f.read()
    raw = np.frombuffer(pixels, dtype = np.float)
    raw = np.reshape(raw, [-1, block_size, block_size, NUM_CHANNELS])
    print(np.shape(raw))
    #print(raw)
    #raw2=raw/255
    #rmean = np.mean(raw[1])
    #print(rmean)
    #raw3=raw2 - rmean
    #print(raw3)  
   
#read labels
with open(label_file, 'r') as f_single_label:
    single_label = f_single_label.read()    
    single_label =np.fromstring (single_label, dtype=np.int32 ,sep=' ')
    single_label = np.reshape(single_label, [-1])
    single_label = keras.utils.to_categorical(single_label, num_classes)
    print(np.shape(single_label))

#read qps
with open(qp_file, 'r') as f_qp:
    qps = f_qp.read()
    qps =np.fromstring (qps, dtype=np.float, sep=' ')
    qps = np.reshape(qps, [-1,1]) 
    print(np.shape(qps)) 


model = cnn16.net()

tbCallBack = TensorBoard(log_dir='./logs',  
                     histogram_freq=0,  
                     write_graph=True,  
                     write_grads=True, 
                     write_images=True,
                     embeddings_freq=0, 
                     embeddings_layer_names=None, 
                     embeddings_metadata=None)

history = model.fit([raw, qps], single_label, validation_split =0.1, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[tbCallBack])

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

model.save('model/intra/16/my_model.h5') 

'''

model=load_model('my_model.h5') 

tsample_file = 'training_samples_all_intra_16.txt'
tlabel_file = 'labels_16_intra.txt'
tqp_file = 'qps_16_intra.txt'

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


classes = model.predict([traw, tqps])

print(classes)

'''
