import os, sys, pickle, json
from os import makedirs
from os.path import exists, join
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard
import keras_model_16 as cnn16
import keras_model_32 as cnn32
import keras_model_64 as cnn64
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels

np.set_printoptions(threshold=sys.maxsize)


batch_size = 64
num_classes = 9
block_size = 64
NUM_CHANNELS = 1
epochs = 100
sample_file = 'mergeanoS_samples_64_intra.txt'
label_file = 'mergeanoS_labels_64_intra.txt'
qp_file = 'mergeanoS_qps_64_intra.txt'
log_dir='./logs/mergeanoS/64/test'

if not exists(log_dir):
    makedirs(log_dir)


class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir=log_dir, **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', 'epoch_'): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()



def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


#read samples
with open(sample_file, 'rb') as f:
    pixels = f.read()
    raw = np.frombuffer(pixels, dtype = np.float)
    raw = np.reshape(raw, [-1, block_size, block_size, NUM_CHANNELS])
    print(np.shape(raw))
    raw_train = raw[:len(raw)//10*9]
    print(np.shape(raw_train))
    raw_test = raw[-len(raw)//10:]
    print(np.shape(raw_test))
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
    y_test = single_label[-len(raw)//10:]
    single_label = keras.utils.to_categorical(single_label, num_classes)
    label_train = single_label[:len(raw)//10*9]
    label_test = single_label[-len(raw)//10:]

    print(np.shape(label_train))
    print(np.shape(label_test))
        
    #print(np.shape(single_label))

#read qps
with open(qp_file, 'r') as f_qp:
    qps = f_qp.read()
    qps =np.fromstring (qps, dtype=np.float, sep=' ')
    qps = np.reshape(qps, [-1,1])
    print(np.shape(qps)) 
    qp_train = qps[:len(raw)//10*9]
    qp_test = qps[-len(raw)//10:]
    print(np.shape(qp_train))
    print(np.shape(qp_test))
  
'''
tbCallBack = TensorBoard(log_dir=log_dir,
                     batch_size=batch_size,                     
                     histogram_freq=0,  
                     write_graph=True,  
                     write_grads=True, 
                     write_images=True,
                     embeddings_freq=0, 
                     embeddings_layer_names=None, 
                     embeddings_metadata=None
                     )
'''
model = cnn64.net()

#model = load_model(log_dir+'/m1_qp120_32_sh.h5')
#class_weight = {0: 8.74, 1: 36.4, 2: 33.82, 3: 1, 4: 132.52, 5: 112.28, 6: 188., 7: 109.24, 8: 63.65, 9: 53.18}
#class_weight = {0: 1.55, 1: 6.87, 2: 7.47, 3: 1, 4: 21.73, 5: 21.2, 6: 23.74, 7: 23.61, 8: 9.64, 9: 11.74} #32
#class_weight = {0: 1., 1: 5.77, 2: 6.29, 3: 11.74, 4: 28.27, 5: 37.52, 6: 28.54, 7: 37.04, 8: 14.1, 9: 15.53} #16

history = model.fit([raw_train, qp_train], label_train, validation_data=([raw_test, qp_test], label_test), batch_size=batch_size, callbacks=[TrainValTensorBoard(write_graph=False)], epochs=epochs, verbose=1)


y_pred = model.predict([raw_test, qp_test])
y_index =np.zeros(len(y_pred))

for i in range(len(y_pred)):
    y_index[i] = np.argmax(y_pred[i])

#print(y_index)
report = classification_report(y_test,y_index)
print(report)


with open(log_dir+'/precision.txt', 'w') as ps:
    ps.write(report)

y_test = y_test.astype(int)
y_index = y_index.astype(int)

report = classification_report(y_test,y_index, output_dict=True)

with open(log_dir+'/classification_m1_64_anoS', 'wb') as re:
    pickle.dump(report, re)


class_names = ["None", "Horz", "Vert","Horz A", "Horz B", "Vert A", "Vert B", "Horz 4", "Vert 4"]
print('Confusion Matrix')

plot_confusion_matrix(y_test, y_index, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.show()

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(log_dir+'/m1_qp120_64_acc_anoS.jpg')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(log_dir+'/m1_qp120_64_loss_anoS.jpg')
plt.show()



with open(log_dir+'/trainHistoryDict_m1_64_anoS', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)


model.save(log_dir+'/m1_qp120_64_anoS.h5') 

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
