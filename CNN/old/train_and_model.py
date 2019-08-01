from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, time
import numpy as np  
import tensorflow as tf  



assert len(sys.argv) == 6
sample_file = sys.argv[1]
label_file = sys.argv[2]
qp_file = sys.argv[3]
frame_type = int(sys.argv[4])
block_size = int(sys.argv[5])



NUM_CHANNELS = 1 
MAX_STEP = 20000 
BATCH_SIZE = 16
block_size = 16
num_extra = 1 
num_label = 10
learning_rate = 0.001
isdrop = 0
logs_train_dir = 'ckpt/train_and_model/16/16_0.001'
checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
     

# weight initialization
def weight_variable(shape, name=None):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name=name)
 	
# convolution
def activate(x, acti_mode):
	if acti_mode==0:
		return x
	elif acti_mode==1:
		return tf.nn.leaky_relu(x)
	elif acti_mode==2:
		return tf.nn.sigmoid(x)
	elif acti_mode==3:
		return tf.nn.softmax(x)

def aver_pool(x, k_width):
	return tf.nn.avg_pool(x, ksize=[1, k_width, k_width, 1], strides=[1, k_width, k_width, 1], padding='SAME')
    

def zero_mean_norm_local(x, x_width, kernel_width):
	w_norm = tf.constant(1.0/(kernel_width*kernel_width), tf.float32, shape=[kernel_width, kernel_width,1,1])
	x_mean_reduced = tf.nn.conv2d(x, w_norm, [1, kernel_width, kernel_width, 1],'VALID')
	x_mean_expanded = tf.image.resize_nearest_neighbor(x_mean_reduced, [x_width, x_width])
	return x-x_mean_expanded

def non_overlap_conv(x, k_width, num_filters_in, num_filters_out, acti_mode, l_name=None):
	with tf.name_scope(l_name):
		w_conv = weight_variable([k_width, k_width, num_filters_in, num_filters_out])
		b_conv = bias_variable([num_filters_out])
		tf.summary.histogram('weights', w_conv)
		tf.summary.histogram('bias', b_conv)
		h_conv = tf.nn.conv2d(x, w_conv, strides=[1, k_width, k_width, 1], padding='VALID') + b_conv
		tf.summary.histogram('pre_activations', h_conv)
		h_conv_a = activate(h_conv, acti_mode)
		tf.summary.histogram('activations', h_conv_a)
	return(h_conv_a)    

def full_connect(x, num_filters_in, num_filters_out, acti_mode, rate=0, name_l=None):
	with tf.name_scope(name_l):
		w_fc = weight_variable([num_filters_in, num_filters_out])
		b_fc = bias_variable([num_filters_out])
		tf.summary.histogram('weights', w_fc)
		tf.summary.histogram('bias', b_fc)
		h_fc = tf.matmul(x, w_fc) + b_fc
		tf.summary.histogram('pre_activations', h_fc)
		h_fc_a = activate(h_fc, acti_mode)
		tf.summary.histogram('activations', h_fc_a)
		h_fc_a = tf.cond(rate > tf.constant(0.0), lambda: tf.nn.dropout(h_fc_a, rate=rate), lambda: h_fc_a)
		#h_fc = tf.nn.dropout(h_fc, rate=1-keep_prob)
	return h_fc_a


num_conv1_filter = 16
num_conv2_filter = 24
num_conv3_filter = 32

num_conv3_concat = 1 * 1 * num_conv3_filter 
num_full_con = 16   


with tf.name_scope('input'):
	x = tf.placeholder("float", [None, block_size, block_size, NUM_CHANNELS], name= 'raw-image')
	qp = tf.placeholder("float",[None, 1])
	y = tf.placeholder("int32",[None], name='label')
	#ys = tf.placeholder("float",[None,10])
	drop = tf.placeholder("float") #need to check this
	x = tf.scalar_mul(1.0 / 255.0, x)
	qp = tf.scalar_mul(1 / 255.0, qp)
	#qp = tf.reshape(qp, [-1,1])


with tf.name_scope('input_reshape'):
	x_image = tf.reshape(x, [-1, block_size, block_size, NUM_CHANNELS])
	image_m = zero_mean_norm_local(x_image, 16, 16)
	tf.summary.image('input', image_m, 10)


conv1 = non_overlap_conv(image_m, 4, NUM_CHANNELS, num_conv1_filter, acti_mode = 1, l_name='conv1') #out batch*4*4*16
conv2 = non_overlap_conv(conv1, 2, num_conv1_filter, num_conv2_filter, acti_mode = 1, l_name='conv2') # out batch*2*2*24
conv3 = non_overlap_conv(conv2, 2, num_conv2_filter, num_conv3_filter, acti_mode = 1, l_name='conv3') # out batch*1*1*32

conv3_flat = tf.reshape(conv3, [-1, num_conv3_concat])
conv3_flat_c = tf.concat([conv3_flat, qp], axis=1)

fc_1 = full_connect(conv3_flat_c, num_conv3_concat + num_extra, num_full_con, acti_mode=1, rate=drop*0.5, name_l='fc1')
fc_1_c = tf.concat([fc_1, qp], axis=1)

prob = full_connect(fc_1_c, num_full_con + num_extra, num_label, acti_mode = 0 , name_l='output')

#opt_vars_all_64 = [v for v in tf.trainable_variables()]


with tf.variable_scope('loss') as scope:  
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prob+1e-8, labels=y, name='xentropy_per_example')
	loss = tf.reduce_mean(cross_entropy, name='loss')  
	tf.summary.scalar(scope.name + '/cross_entropy', loss)  


with tf.name_scope('train'):
	optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)  
	global_step = tf.Variable(0, name='global_step', trainable=False)  
	train_op = optimizer.minimize(loss, global_step = global_step)  


with tf.variable_scope('accuracy') as scope:
	correct = tf.nn.in_top_k(prob, y, 1)  
	correct = tf.cast(correct, tf.float16)  
	accuracy = tf.reduce_mean(correct)
	tf.summary.scalar(scope.name + '/accuracy', accuracy)


sess = tf.Session()
sess.run(tf.global_variables_initializer())


saver = tf.train.Saver()
#saver.restore(sess, tf.train.latest_checkpoint(logs_train_dir))

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


for step in range(number):
	if step % 100 == 0:
		summary_str,_, tra_loss, tra_acc = sess.run([summary_op, train_op, loss, accuracy], feed_dict={x: data[step*BATCH_SIZE:((step+1)*BATCH_SIZE), :], qp: qps[step*BATCH_SIZE:((step+1)*BATCH_SIZE),:], drop: isdrop,  y: single_label[step*BATCH_SIZE:((step+1)*BATCH_SIZE)]})       
		train_writer.add_summary(summary_str, step+70800)
		saver.save(sess, checkpoint_path, global_step=step+70800)  
		print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100))
	else:
		_, tra_loss, tra_acc = sess.run([train_op, loss, accuracy], feed_dict={x: data[step*BATCH_SIZE:((step+1)*BATCH_SIZE), :], qp: qps[step*BATCH_SIZE:((step+1)*BATCH_SIZE),:], drop: isdrop,  y: single_label[step*BATCH_SIZE:((step+1)*BATCH_SIZE)]})
	#if step % 10 == 0 or (step + 1) == MAX_STEP:
	#	checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
	#	saver.save(sess, checkpoint_path, global_step=step)   

sess.close()  
