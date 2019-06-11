from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf

IMAGE_SIZE = 32
NUM_CHANNELS = 1 

NUM_EXT_FEATURES = 1 
NUM_LABEL_BYTES = 10

NUM_CONVLAYER1_FILTERS = 16
NUM_CONVLAYER2_FILTERS = 24
NUM_CONVLAYER3_FILTERS = 32

NUM_CONV2_FLAT_FILTERS = 2 * 2 * NUM_CONVLAYER2_FILTERS
NUM_CONV3_FLAT_FILTERS = 1 * 1 * NUM_CONVLAYER3_FILTERS 

NUM_CONVLAYER_FLAT_FILTERS = NUM_CONV2_FLAT_FILTERS + NUM_CONV3_FLAT_FILTERS

NUM_DENLAYER1_FEATURES_32 = 96        
NUM_DENLAYER2_FEATURES_32 = 64


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
    print(x_mean_reduced)
    print(x_mean_expanded)
    return x-x_mean_expanded

def non_overlap_conv(x, k_width, num_filters_in, num_filters_out, acti_mode):
    w_conv = weight_variable([k_width, k_width, num_filters_in, num_filters_out])
    b_conv = bias_variable([num_filters_out])
    h_conv = tf.nn.conv2d(x, w_conv, strides=[1, k_width, k_width, 1], padding='VALID') + b_conv
    h_conv = activate(h_conv, acti_mode)
    print(h_conv)
    return(h_conv)    

def full_connect(x, num_filters_in, num_filters_out, acti_mode, keep_prob=1, name_w=None, name_b=None): 
    w_fc = weight_variable([num_filters_in, num_filters_out], name_w)
    b_fc = bias_variable([num_filters_out], name_b)
    h_fc = tf.matmul(x, w_fc) + b_fc
    h_fc = activate(h_fc, acti_mode)
    h_fc = tf.cond(keep_prob < tf.constant(1.0), lambda: tf.nn.dropout(h_fc, keep_prob), lambda: h_fc)
    print(h_fc) 
    return h_fc

def net(x, qp, isdrop):
    
    x = tf.scalar_mul(1.0 / 255.0, x)
    qp = tf.scalar_mul(1 / 255.0, qp)
    #x_image = tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
    #qp = tf.reshape(qp, [-1,1])
    #----------------------------------------------------------------------------------------------------------------------------------------------------------
    acti_mode_conv = 1
    
    # for extracting textures of 32*32 CTUs
    #h_image_L = zero_mean_norm_local(x, 32, 32)
    h_image_L = zero_mean_norm_local(aver_pool(x, 2), 16, 16)
    h_conv1_L = non_overlap_conv(h_image_L, 4, NUM_CHANNELS, NUM_CONVLAYER1_FILTERS, acti_mode = acti_mode_conv) #out batch, 16*16 16
    h_conv2_L = non_overlap_conv(h_conv1_L, 2, NUM_CONVLAYER1_FILTERS, NUM_CONVLAYER2_FILTERS, acti_mode = acti_mode_conv) # out batch, 8*8  16*24
    h_conv3_L = non_overlap_conv(h_conv2_L, 2, NUM_CONVLAYER2_FILTERS, NUM_CONVLAYER3_FILTERS, acti_mode = acti_mode_conv) # out batch, 4*4 16*24*32 
   
    h_conv2_flat = tf.reshape(h_conv2_L, [-1, NUM_CONV2_FLAT_FILTERS])
    h_conv3_flat = tf.reshape(h_conv3_L, [-1, NUM_CONV3_FLAT_FILTERS])
    
    h_conv_flat = tf.concat( values=[h_conv3_flat, h_conv2_flat], axis=1)
    print(h_conv_flat)

    acti_mode_fc = 1
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------

    h_fc1_32 = full_connect(h_conv_flat, NUM_CONVLAYER_FLAT_FILTERS, NUM_DENLAYER1_FEATURES_32, 
                            acti_mode=acti_mode_fc, keep_prob=1-isdrop*0.5, name_w='h_fc1__32__w', name_b='h_fc1__32__b')
    h_fc1_32_c = tf.concat([h_fc1_32, qp], axis=1)
    h_fc2_32 = full_connect(h_fc1_32_c, NUM_DENLAYER1_FEATURES_32 + NUM_EXT_FEATURES, NUM_DENLAYER2_FEATURES_32, 
                            acti_mode=acti_mode_fc, keep_prob=1-isdrop*0.2, name_w='h_fc2__32__w', name_b='h_fc2__32__b')    
    h_fc2_32_c = tf.concat([h_fc2_32, qp], axis=1)
    y_conv_flat_32 = full_connect(h_fc2_32_c, NUM_DENLAYER2_FEATURES_32 + NUM_EXT_FEATURES, 10, 
                            acti_mode = 0 , name_w='y_conv_flat__32__w', name_b='y_conv_flat__32__b')

    opt_vars_all_32 = [v for v in tf.trainable_variables()]
    
    #do we need to declair and give a array first? no need, it is automaticly adopt the shape from top input
    #partition = np.argmax(y_conv_flat_64) here it returns the whole predictions results, 10 probabilities
    #and let the evaluation to pick the label 


    return y_conv_flat_32, opt_vars_all_32


# every sample only has one label 
#this part use softmax function, that's why the output of net doesn't use softmax. 
# in the model every layer uses leaky_Relu, only the final output uses sigmoid
#but for evaluation, softmax is not really necessary, as long as the biggst number is 
def losses(logits, labels):   
    with tf.variable_scope('loss') as scope:  
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits \
                        (logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')  
        tf.summary.scalar(scope.name + '/cross_entropy', loss)  
    return loss     

def trainning(loss, learning_rate):  
    with tf.name_scope('optimizer'):  #this means optimizer = tf.name_scope
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