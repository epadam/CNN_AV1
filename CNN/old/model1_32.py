from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf

block_size = 32

NUM_CHANNELS = 1 

num_extra = 1 
num_label = 10

num_conv1_filter = 16
num_conv2_filter = 24
num_conv3_filter = 32

num_conv2_concat = 2 * 2 * num_conv2_filter
num_conv3_concat = 1 * 1 * num_conv3_filter 

num_concat_all = num_conv2_concat + num_conv3_concat

num_full_con1 = 64
num_full_con2 = 48       



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
        h_fc_a = activate(h_fc, acti_mode)
        tf.summary.histogram('activations', h_fc_a)
        h_fc_a = tf.cond(rate > tf.constant(0.0), lambda: tf.nn.dropout(h_fc_a, rate=rate), lambda: h_fc_a)    
        #h_fc = tf.nn.dropout(h_fc, rate=1-keep_prob)
    return h_fc

def net(x, qp, drop):

    with tf.name_scope('input_reshape'):
        x = tf.scalar_mul(1.0 / 255.0, x)
        qp = tf.scalar_mul(1 / 255.0, qp)
        x_image = tf.reshape(x, [-1, block_size, block_size, NUM_CHANNELS])
        image_m = zero_mean_norm_local(aver_pool(x_image, 2), 16, 16)
        tf.summary.image('input', image_m, 10)

    conv1 = non_overlap_conv(image_m, 4, NUM_CHANNELS, num_conv1_filter, acti_mode = 1, l_name='conv1') #out batch*4*4*16
    conv2 = non_overlap_conv(conv1, 2, num_conv1_filter, num_conv2_filter, acti_mode = 1, l_name='conv2') # out batch*2*2*24
    conv3 = non_overlap_conv(conv2, 2, num_conv2_filter, num_conv3_filter, acti_mode = 1, l_name='conv3') # out batch*1*1*32

    conv2_flat=tf.reshape(conv2, [-1, num_conv2_concat])   
    conv3_flat = tf.reshape(conv3, [-1, num_conv3_concat])
    conv_flat_all = tf.concat( values=[conv3_flat, conv2_flat], axis=1)

    fc_1 = full_connect(conv_flat_all, num_concat_all, num_full_con1, acti_mode=1, rate=drop*0.5, name_l='fc1')
    fc_1_c = tf.concat([fc_1, qp], axis=1)

    fc_2 = full_connect(fc_1_c, num_full_con1 + num_extra, num_full_con2, acti_mode=1, rate=drop*0.2, name_l='fc2')
    fc_2_c = tf.concat([fc_2, qp], axis=1)


    prob = full_connect(fc_2_c, num_full_con2 + num_extra, num_label, acti_mode = 0 , name_l='model_output')

    #opt_vars_all_64 = [v for v in tf.trainable_variables()]

    return prob

#1.weighted cross entropy, labels needs to be 10(ys not y), output no softmax or sigmoid
'''
def losses(logits, labels):   
    with tf.variable_scope('loss') as scope:
        classes_weights = tf.constant([0.03727, 1.1587, 2.4038, 0.0000687, 4.6511, 4.0505, 8.1967, 7.0921, 2.6246, 8.6956])
        cross_entropy = tf.nn.weighted_cross_entropy_with_logits(targets= labels, logits = logits, pos_weight = classes_weights, name= 'xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')  
        tf.summary.scalar(scope.name + '/cross_entropy', loss)  
    return loss
'''
#2.change output mode to softmax, also ys not y
'''
def losses(logits, labels):   
    with tf.variable_scope('loss') as scope:
        coe = tf.constant([0.1, 10.0, 10.0, 0.01, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        y_coe = labels*coe
        loss = -tf.reduce_mean(y_coe*tf.log(logits)) 
        tf.summary.scalar(scope.name + '/cross_entropy', loss)  
    return loss       
'''

#3.original, but use even data
def losses(logits, labels):   
    with tf.variable_scope('loss') as scope:  
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xentropy')
        loss = tf.reduce_mean(cross_entropy, name='loss')  
        tf.summary.scalar(scope.name + '/cross_entropy', loss)  
    return loss 



def trainning(loss, learning_rate):  
    with tf.name_scope('training'):  #this means optimizer = tf.name_scope
        optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)  
        global_step = tf.Variable(0, name='global_step', trainable=False)  
        train_op = optimizer.minimize(loss, global_step= global_step)  
    return train_op 

#only one hot label, ex: if there are  logits:label is 10:1
def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:   #this means accuracy = tf.variable_scope and as name scope?
        correct = tf.nn.in_top_k(logits, labels, 1)  
        correct = tf.cast(correct, tf.float16)  
        accuracy = tf.reduce_mean(correct)  
        tf.summary.scalar(scope.name + '/accuracy', accuracy)  
    return accuracy
