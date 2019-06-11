from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys,shutil
import numpy as np
import random
import net_CNN_64 as nt64
import net_CNN_32 as nt32
import net_CNN_16 as nt16
import math
import tensorflow as tf
import matplotlib.pyplot as plt

import time

os.environ['CUDA_VISIBLE_DEVICES'] = '' # only CPU is enabled

NUM_CHANNELS = 1
NUM_EXT_FEATURES = 1
SAVE_FILE = 'cu_depth.txt'


def print_current_line(str):
    print('\r' + str, end = '')
    sys.stdout.flush()

def print_clear():
    print('\r', end = '')
    sys.stdout.flush()


def get_Y_for_one_frame(f, frame_width, frame_height, image_size, frame_number):
    f.seek(frame_number*(frame_width * frame_height + frame_width * frame_height//2), 0)
    y_buf = f.read(frame_width * frame_height)
    data = np.frombuffer(y_buf, dtype = np.uint8)
    data = data.reshape(frame_height, frame_width)

    valid_height = math.ceil(frame_height / image_size) * image_size
    valid_width = math.ceil(frame_width / image_size) * image_size
    if valid_height > frame_height:
        data = np.concatenate((data, np.zeros((valid_height - frame_height, frame_width))), axis = 0)
    if valid_width > frame_width:
        data = np.concatenate((data, np.zeros((valid_height, valid_width - frame_width))), axis = 1)
      
    return data
 
def get_y_conv_for_one_frame(input_image, image_size, qp_one_frame):

    if image_size == 64:
        #NUM_CONV2_FLAT_FILTERS = 8*8*24
        #NUM_CONV3_FLAT_FILTERS = 4*4*32
        if frame_type == 0:
            logs_train_dir = 'SaveNet/intra/64'
        else:
            logs_train_dir = 'SaveNet/inter/64'

        x = tf.placeholder("float", [None, image_size, image_size, NUM_CHANNELS])
        qp = tf.placeholder("float", [None, NUM_EXT_FEATURES])
        isdrop = tf.placeholder("float")
        y_conv_flat, opt_vars_all_64 = nt64.net(x, qp, isdrop)
        sess = tf.Session()
        tf.reset_default_graph()
        saver = tf.train.Saver(opt_vars_all_64, write_version = tf.train.SaverDef.V2)
        ckpt = tf.train.get_checkpoint_state(logs_train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('success, training step= %s' % global_step)
    
    elif image_size == 32:
        #NUM_CONV2_FLAT_FILTERS = 4*4*24
        #NUM_CONV3_FLAT_FILTERS = 2*2*32

        if frame_type == 0:
            logs_train_dir = 'SaveNet/intra/32'
        else:
            logs_train_dir = 'SaveNet/inter/32'

        x = tf.placeholder("float", [None, image_size, image_size, NUM_CHANNELS])
        qp = tf.placeholder("float", [None, NUM_EXT_FEATURES])
        isdrop = tf.placeholder("float")
        y_conv_flat, opt_vars_all_32 = nt32.net(x, qp, isdrop) 
        sess = tf.Session()
        tf.reset_default_graph()
        saver = tf.train.Saver(opt_vars_all_32, write_version = tf.train.SaverDef.V2)
        ckpt = tf.train.get_checkpoint_state(logs_train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('success, training step= %s' % global_step)
    elif image_size ==16:
        #NUM_CONV2_FLAT_FILTERS = 2*2*24
        #NUM_CONV3_FLAT_FILTERS = 1*1*32
        
        if frame_type == 0:
            logs_train_dir = 'SaveNet/intra/16'
        else:
            logs_train_dir = 'SaveNet/inter/16'

        x = tf.placeholder("float", [None, image_size, image_size, NUM_CHANNELS])
        qp = tf.placeholder("float", [None, NUM_EXT_FEATURES])
        isdrop = tf.placeholder("float")
        y_conv_flat, opt_vars_all_16 = nt16.net(x, qp, isdrop) 

        sess = tf.Session()
        tf.reset_default_graph()
        saver = tf.train.Saver(opt_vars_all_16, write_version = tf.train.SaverDef.V2)
        ckpt = tf.train.get_checkpoint_state(logs_train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('success, training step= %s' % global_step)   

    batch_size = np.shape(input_image)[0] #how many blocks needed to be predicted
    y_flat_temp = np.zeros((batch_size, 10))
    sub_batch_size = 1024

    
    for i in range(batch_size):
        y_flat_temp [i] = sess.run(y_conv_flat, feed_dict={x: input_image[i:i+1, :].astype(np.float32), qp: (np.ones((1, NUM_EXT_FEATURES)) * qp_one_frame).astype(np.float32), isdrop: 0})
        #print(y_flat_temp [i])
        image_show = input_image[i].reshape(image_size, image_size)
        #plt.imshow(image_show)
        #plt.colorbar()
        #plt.show()
    '''    
    for i in range(math.ceil(batch_size / sub_batch_size)):
        index_start = i * sub_batch_size
        index_end = (i + 1) * sub_batch_size
        if index_end > batch_size:
            index_end = batch_size       
        y_flat_temp [index_start:index_end] = sess.run(y_conv_flat, feed_dict={x: input_image[index_start:index_end, :].astype(np.float32), qp: (np.ones((index_end - index_start, NUM_EXT_FEATURES)) * qp_one_frame).astype(np.float32), isdrop: 0})
        #print(y_flat_temp [index_start:index_end])
    '''
    return y_flat_temp
 

def prob_64(yuv_name, save_file, qp_one_frame, frame_width, frame_height, frame_number, frame_type):

    image_size = 64
    
    f_out = open(save_file, 'w+')
    
    f = open(yuv_name, 'rb') #need read different frames
    valid_luma = get_Y_for_one_frame(f, frame_width, frame_height, image_size, frame_number)

    #calculate how many samples
    valid_height = math.ceil(frame_height / image_size) * image_size
    valid_width = math.ceil(frame_width / image_size) * image_size
    #batch_size = (valid_height // image_size) * (valid_width // image_size)
    batch_size = math.ceil(frame_height / image_size) * math.ceil(frame_width / image_size)
    input_batch = np.zeros((batch_size, image_size, image_size, NUM_CHANNELS))
    
    prob = np.zeros(batch_size) # has number of 64*64 blocks in the frame   
    
    #each sample has one label
    #partition = np.zeros(batch_size)
        
    index = 0
    ystart = 0
    while ystart < valid_height:
        xstart = 0
        while xstart < valid_width:
            CU_input = valid_luma[ystart : ystart + image_size, xstart : xstart + image_size]
            input_batch[index] = np.reshape(CU_input, [1, image_size, image_size, NUM_CHANNELS])
            index += 1
            xstart += image_size
        ystart += image_size  
            
    input_batch = input_batch.astype(np.float32)
    partition_output = get_y_conv_for_one_frame(input_batch, image_size, qp_one_frame)

    for i in range(batch_size):
        print(partition_output[i])
        prob[i] = np.argmax(partition_output[i])

    #prob[0 : batch_size] = partition
    
    prob = prob.astype(int)
    
    print(prob)
    #prob_arr = np.reshape(prob.astype(str), [1, (valid_height // image_size) * (valid_width // image_size) )])
    
    prob_arr= "".join(prob.astype(str))
    
    f_out.write(prob_arr) 
    f_out.close()

def prob_32(yuv_name, save_file, qp_one_frame, frame_width, frame_height, frame_number, frame_type):
    image_size = 32
    
    f_out = open(save_file, 'a')

    f = open(yuv_name, 'rb') #need read different frames
    valid_luma = get_Y_for_one_frame(f, frame_width, frame_height, image_size, frame_number)

    batch_size = math.ceil(frame_height / image_size) * math.ceil(frame_width / image_size)
    input_batch = np.zeros((batch_size, image_size, image_size, NUM_CHANNELS))
    
    prob = np.zeros(batch_size)
        
    index = 0
    ystart = 0
    while ystart < frame_height:
        xstart = 0
        while xstart < frame_width:
            CU_input = valid_luma[ystart : ystart + image_size, xstart : xstart + image_size]
            input_batch[index] = np.reshape(CU_input, [1, image_size, image_size, NUM_CHANNELS])
            index += 1
            xstart += image_size
        ystart += image_size  
            
    input_batch = input_batch.astype(np.float32)
    partition_output = get_y_conv_for_one_frame(input_batch, image_size, qp_one_frame)

    for i in range(batch_size):
        #print(partition_output[i])
        prob[i] = np.argmax(partition_output[i])

    #prob[0 : batch_size] = partition
    
    prob = prob.astype(int)
    
    print(prob)
    #prob_arr = np.reshape(prob.astype(str), [1, (valid_height // image_size) * (valid_width // image_size) )])
    
    prob_arr= "".join(prob.astype(str))
    
    f_out.write(prob_arr) 
    f_out.close()


def prob_16(yuv_name, save_file, qp_one_frame, frame_width, frame_height, frame_number, frame_type):

    image_size = 16
    f_out = open(save_file, 'a')

    f = open(yuv_name, 'rb') #need read different frames
    valid_luma = get_Y_for_one_frame(f, frame_width, frame_height, image_size, frame_number)

    batch_size = math.ceil(frame_height / image_size) * math.ceil(frame_width / image_size)
    input_batch = np.zeros((batch_size, image_size, image_size, NUM_CHANNELS))
    
    prob = np.zeros(batch_size)
        
    index = 0
    ystart = 0
    while ystart < frame_height:
        xstart = 0
        while xstart < frame_width:
            CU_input = valid_luma[ystart : ystart + image_size, xstart : xstart + image_size]
            input_batch[index] = np.reshape(CU_input, [1, image_size, image_size, NUM_CHANNELS])
            index += 1
            xstart += image_size
        ystart += image_size  

    input_batch = input_batch.astype(np.float32)
    partition_output = get_y_conv_for_one_frame(input_batch, image_size, qp_one_frame)

    for i in range(batch_size):
        #print(partition_output[i])
        prob[i] = np.argmax(partition_output[i])

    #prob[0 : batch_size] = partition
    
    prob = prob.astype(int)
    
    print(prob)
    #prob_arr = np.reshape(prob.astype(str), [1, (valid_height // image_size) * (valid_width // image_size) )])
    
    prob_arr= "".join(prob.astype(str))
    
    f_out.write(prob_arr) 
    f_out.close()



assert len(sys.argv) == 7
yuv_file = sys.argv[1]
width = int(sys.argv[2])
height = int(sys.argv[3])
qp_one_frame = int(sys.argv[4])
frame_type = int(sys.argv[5])
frame_number = int(sys.argv[6])



t1 = time.time()
prob_64(yuv_file, SAVE_FILE, qp_one_frame, width, height, frame_number, frame_type)
#prob_32(yuv_file, SAVE_FILE, qp_one_frame, width, height, frame_number, frame_type)
#prob_16(yuv_file, SAVE_FILE, qp_one_frame, width, height, frame_number, frame_type)
t2 = time.time()
print('--------\n\nPredicting Time: %.3f sec.\n\n--------' % float(t2-t1))

