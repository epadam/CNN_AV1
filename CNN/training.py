import os, sys, time
import numpy as np  
import tensorflow as tf  
import model1_64 as model


NUM_CHANNELS = 1 
NUM_EXT_FEATURES = 1 
N_CLASSES = 10  
MAX_STEP = 20000 
BATCH_SIZE = 64
drop_rate = 0
learning_rate = 0.001
logs_train_dir = 'ckpt/16/1080_m1_16_0.001'
checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')

def run_training(sample_file, label_file, qp_file, frame_type, block_size):
    
    with tf.name_scope('input'):
        x = tf.placeholder("float", [None, block_size, block_size, NUM_CHANNELS], name= 'raw-image')
        qp = tf.placeholder("float",[None, 1], name='qp')
        y = tf.placeholder("int32",[None], name='label')
        #ys = tf.placeholder("float",[None,10])																																																																																																																																																																																																																			
        drop = tf.placeholder("float", name='droprate') #need to check this


    train_logits = model.net(x, qp, drop)
    loss = model.losses(train_logits, y)
    train_op = model.trainning(loss, learning_rate)
    accuracy = model.evaluation(train_logits, y)
    
    sess = tf.Session()
    saver = tf.train.Saver()
    
    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    
    

    #read samples
    with open(sample_file, 'rb') as f:
        pixels = f.read()
        data = np.frombuffer(pixels, dtype = np.float)
        data = np.reshape(data, [-1, block_size, block_size, NUM_CHANNELS])
        print(np.shape(data))  
       
    #read labels
    with open(label_file, 'r') as f_single_label:
        f_single_label = open(label_file, 'r')
        single_label = f_single_label.read()    
        single_label =np.fromstring (single_label, dtype=np.int32 ,sep=' ')
        single_label = np.reshape(single_label, [-1]) 
        print(np.shape(single_label))

    #labels_10 = np.loadtxt("labels_10_64.txt", dtype=float)

    #read qps
    with open(qp_file, 'r') as f_qp:
        qps = f_qp.read()
        qps =np.fromstring (qps, dtype=np.float, sep=' ')
        qps = np.reshape(qps, [-1,1]) 
        print(np.shape(qps)) 
    

    number = int(np.size(qps)/BATCH_SIZE)
    print(number)
 
    sess.run(tf.global_variables_initializer())

    #saver.restore(sess, tf.train.latest_checkpoint(logs_train_dir))


    for step in range(number):
        if step % 100 == 0:
            summary_str,_, tra_loss, tra_acc = sess.run([summary_op, train_op, loss, accuracy], feed_dict={x: data[step*BATCH_SIZE:((step+1)*BATCH_SIZE), :], qp: qps[step*BATCH_SIZE:((step+1)*BATCH_SIZE),:], drop: drop_rate,  y: single_label[step*BATCH_SIZE:((step+1)*BATCH_SIZE)]})       
            train_writer.add_summary(summary_str, step)
            checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step) 
            print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100))
        else:
            _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy], feed_dict={x: data[step*BATCH_SIZE:((step+1)*BATCH_SIZE), :], qp: qps[step*BATCH_SIZE:((step+1)*BATCH_SIZE),:], drop: drop_rate,  y: single_label[step*BATCH_SIZE:((step+1)*BATCH_SIZE)]})
        #if step % 100 == 0 or (step + 1) == MAX_STEP:
            #checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
            #saver.save(sess, checkpoint_path, global_step=step)   

    sess.close() 


# train
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

 
