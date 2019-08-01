import os, sys, time
import numpy as np  
import tensorflow as tf  
import net_CNN_64 as nt64
import net_CNN_32 as nt32
import net_CNN_16 as nt16

NUM_CHANNELS = 1 
NUM_EXT_FEATURES = 1 
N_CLASSES = 10  
MAX_STEP = 20000 
BATCH_SIZE = 64

learning_rate = 0.001

def run_training(sample_file, label_file, qp_file, frame_type, block_size):  
    #model save location
    if frame_type ==0:
        if block_size == 64:
            logs_train_dir = 'SaveNet/intra/64'
        elif block_size == 32:
            logs_train_dir = 'SaveNet/intra/32'
        elif block_size == 16:
            logs_train_dir = 'SaveNet/intra/16'
    else:
        if block_size == 64:
            logs_train_dir = 'SaveNet/inter/64'
        elif block_size == 32:
            logs_train_dir = 'SaveNet/inter/32'
        elif block_size == 16:
            logs_train_dir = 'SaveNet/inter/16'
    '''
    if block_size == 64:
        NUM_CONV2_FLAT_FILTERS = 1536
        NUM_CONV3_FLAT_FILTERS = 512
    elif block_size == 32:
        NUM_CONV2_FLAT_FILTERS = 384
        NUM_CONV3_FLAT_FILTERS = 128
    elif block_size == 16:
        NUM_CONV2_FLAT_FILTERS = 96
        NUM_CONV3_FLAT_FILTERS = 32
    '''

    x = tf.placeholder("float", [None, block_size, block_size, NUM_CHANNELS])
    qp = tf.placeholder("float",[None, 1])
    y = tf.placeholder("int32",[None])
    #ys = tf.placeholder("float",[None,10])
    isdrop = tf.placeholder("float") #need to check this
    
    if block_size == 64:
        train_logits, opt_vars_all_64 = nt64.net(x, qp, isdrop)
        train_loss = nt64.losses(train_logits, y)
        train_op = nt64.trainning(train_loss, learning_rate)
        train__acc = nt64.evaluation(train_logits, y)
        summary_op = tf.summary.merge_all()
        sess = tf.Session()
        train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
        saver = tf.train.Saver(opt_vars_all_64, write_version = tf.train.SaverDef.V2)
    elif block_size == 32:
        train_logits, opt_vars_all_32 = nt32.net(x, qp, isdrop)
        train_loss = nt32.losses(train_logits, y)
        train_op = nt32.trainning(train_loss, learning_rate)
        train__acc = nt32.evaluation(train_logits, y)
        summary_op = tf.summary.merge_all()
        sess = tf.Session()
        train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
        saver = tf.train.Saver(opt_vars_all_32, write_version = tf.train.SaverDef.V2)
    elif block_size ==16:
        train_logits, opt_vars_all_16 = nt16.net(x, qp, isdrop)
        train_loss = nt16.losses(train_logits, y)
        train_op = nt16.trainning(train_loss, learning_rate)
        train__acc = nt16.evaluation(train_logits, y)
        summary_op = tf.summary.merge_all()
        sess = tf.Session()
        train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
        saver = tf.train.Saver(opt_vars_all_16, write_version = tf.train.SaverDef.V2)
    
    #read samples
    f= open(sample_file, 'rb')
    pixels = f.read()
    data = np.frombuffer(pixels, dtype = np.float)
    #data = np.reshape(data, [28, 64*64*NUM_CHANNELS])  
    data = np.reshape(data, [-1, block_size, block_size, NUM_CHANNELS])
    print(np.shape(data))  
       
    #read labels
    f_single_label = open(label_file, 'r')
    single_label = f_single_label.read()    
    single_label =np.fromstring (single_label, dtype=np.int32 ,sep=' ')
    single_label = np.reshape(single_label, [-1]) 
    print(np.shape(single_label))

    #labels_10 = np.loadtxt("labels_10_64.txt", dtype=float)

    #read qps
    f_qp = open(qp_file, 'r') 
    qps = f_qp.read()
    qps =np.fromstring (qps, dtype=np.float, sep=' ')
    qps = np.reshape(qps, [-1,1]) 
    print(np.shape(qps)) 
    
    #can't use the same name as placeholder, otherwise it would show unhashable type

    f.close()
    f_single_label.close()
    f_qp.close()

 
    sess.run(tf.global_variables_initializer())

    '''
    ckpt = tf.train.get_checkpoint_state(logs_train_dir)
    if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('success, training step= %s' % global_step)
    saver.restore(sess, ckpt.model_checkpoint_path)        
    '''



    for step in range(5000):
        summary_str, _, tra_loss, tra_acc = sess.run([summary_op, train_op, train_loss, train__acc], feed_dict={x: data[step*BATCH_SIZE:((step+1)*BATCH_SIZE), :], \
            qp: qps[step*BATCH_SIZE:((step+1)*BATCH_SIZE),:], isdrop: 1,  y: single_label[step*BATCH_SIZE:((step+1)*BATCH_SIZE)]})       
        if step % 10 == 0:  
            print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100))  
            #summary_str = sess.run(summary_op)  need to feed_dict for summary_op
            train_writer.add_summary(summary_str, step)  
              
        if step % 10 == 0 or (step + 1) == MAX_STEP:
            checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)   

    sess.close()  


# train
#input arguments:  inter and intra frame type store 
assert len(sys.argv) == 6
sample_file = sys.argv[1]
label_file = sys.argv[2]
qp_file = sys.argv[3]
frame_type = int(sys.argv[4])
block_size = int(sys.argv[5])

t1 = time.time()
run_training(sample_file, label_file, qp_file, frame_type, block_size)
t2 = time.time()
print('--------\n\nTraining Time: %.3f sec.\n\n--------' % float(t2-t1))

 
