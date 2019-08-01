import os, sys, time, math
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import keras_model_16 as cnn_16
import keras_model_32 as cnn_32
import keras_model_64 as cnn_64

num_classes = 10
block_size = 16
NUM_CHANNELS = 1
SAVE_FILE = 'cu_depth.txt'

#this function searches target frame in the video and add padding according to the block size
def get_blocks_for_one_frame(yuv_name, frame_width, frame_height, block_size, frame_number):
    
    f = open(yuv_name, 'rb')
    #searching frame data in a video
    f.seek(frame_number*(frame_width * frame_height + frame_width * frame_height//2), 0) 
    #only read luma data
    y_buf = f.read(frame_width * frame_height)
    data = np.frombuffer(y_buf, dtype = np.uint8)
    data = data.reshape(frame_height, frame_width)
    
    #padding the frame according to the block size
    valid_height = math.ceil(frame_height / block_size) * block_size
    valid_width = math.ceil(frame_width / block_size) * block_size
    if valid_height > frame_height:
        data = np.concatenate((data, np.zeros((valid_height - frame_height, frame_width))), axis = 0)
    if valid_width > frame_width:
        data = np.concatenate((data, np.zeros((valid_height, valid_width - frame_width))), axis = 1)
      
    return data

#this function predicts the partition mode for each block in the frame  
def get_partition_for_one_frame(blocks, block_size, qps, frame_type):

    
    if block_size == 64:
        model = cnn_64.net() 
        if frame_type == 0:
            model=load_model('model/intra/64')
        else:
            model=load_model('model/inter/64')
    elif block_size == 32:
        model = cnn_32.net() 
        if frame_type == 0:
            model=load_model('model/intra/32')
        else:
            model=load_model('model/inter/32')
    else:
        model = cnn_16.net() 
        if frame_type == 0:
            model=load_model('model/intra/16/my_model.h5')
        else:
            model=load_model('model/inter/16')

    return model.predict([blocks, qps])



def prob_64(yuv_name, save_file, qp_one_frame, frame_width, frame_height, frame_number, frame_type):

    block_size = 64
    #get raw image with padding
    luma_data = get_blocks_for_one_frame(yuv_name, frame_width, frame_height, block_size, frame_number)

    num_samples = math.ceil(frame_height / block_size) * math.ceil(frame_width / block_size)
    blocks = np.zeros((num_samples, block_size, block_size, NUM_CHANNELS))
    qps = np.full((num_samples, 1), qp_one_frame)
    
    #rearrange image into block-wise    
    index = 0
    ystart = 0
    while ystart < frame_height:
        xstart = 0
        while xstart < frame_width:
            one_block = luma_data[ystart : ystart + block_size, xstart : xstart + block_size]
            blocks[index] = np.reshape(one_block, [1, block_size, block_size, NUM_CHANNELS])
            index += 1
            xstart += block_size
        ystart += block_size  

    blocks = blocks.astype(np.float32)
    #partition_output returns array with shape [num_samples, classes]
    partition_output = get_partition_for_one_frame(blocks, block_size, qps, frame_type)

    prob = np.zeros(num_samples)

    for i in range(num_samples):
        #print(partition_output[i])
        prob[i] = np.argmax(partition_output[i])
    
    prob = prob.astype(int)
    
    print(prob)
    #prob_arr = np.reshape(prob.astype(str), [1, (valid_height // block_size) * (valid_width // block_size) )])
    
    prob_str= "".join(prob.astype(str))
    
    with open(save_file, 'a') as f_out:
        f_out.write(prob_str)


def prob_32(yuv_name, save_file, qp_one_frame, frame_width, frame_height, frame_number, frame_type):

    block_size = 32
    #get raw image with padding
    luma_data = get_blocks_for_one_frame(yuv_name, frame_width, frame_height, block_size, frame_number)

    num_samples = math.ceil(frame_height / block_size) * math.ceil(frame_width / block_size)
    blocks = np.zeros((num_samples, block_size, block_size, NUM_CHANNELS))
    qps = np.full((num_samples, 1), qp_one_frame)
    
    #rearrange image into block-wise    
    index = 0
    ystart = 0
    while ystart < frame_height:
        xstart = 0
        while xstart < frame_width:
            one_block = luma_data[ystart : ystart + block_size, xstart : xstart + block_size]
            blocks[index] = np.reshape(one_block, [1, block_size, block_size, NUM_CHANNELS])
            index += 1
            xstart += block_size
        ystart += block_size  

    blocks = blocks.astype(np.float32)
    #partition_output returns array with shape [num_samples, classes]
    partition_output = get_partition_for_one_frame(blocks, block_size, qps, frame_type)

    prob = np.zeros(num_samples)

    for i in range(num_samples):
        #print(partition_output[i])
        prob[i] = np.argmax(partition_output[i])
    
    prob = prob.astype(int)
    
    print(prob)
    #prob_arr = np.reshape(prob.astype(str), [1, (valid_height // block_size) * (valid_width // block_size) )])
    
    prob_str= "".join(prob.astype(str))
    
    with open(save_file, 'a') as f_out:
        f_out.write(prob_str)


def prob_16(yuv_name, save_file, qp_one_frame, frame_width, frame_height, frame_number, frame_type):

    block_size = 16
    #get raw image with padding
    luma_data = get_blocks_for_one_frame(yuv_name, frame_width, frame_height, block_size, frame_number)

    num_samples = math.ceil(frame_height / block_size) * math.ceil(frame_width / block_size)
    blocks = np.zeros((num_samples, block_size, block_size, NUM_CHANNELS))
    qps = np.full((num_samples, 1), qp_one_frame)
    
    #rearrange image into block-wise    
    index = 0
    ystart = 0
    while ystart < frame_height:
        xstart = 0
        while xstart < frame_width:
            one_block = luma_data[ystart : ystart + block_size, xstart : xstart + block_size]
            blocks[index] = np.reshape(one_block, [1, block_size, block_size, NUM_CHANNELS])
            index += 1
            xstart += block_size
        ystart += block_size  

    blocks = blocks.astype(np.float32)
    #partition_output returns array with shape [num_samples, classes]
    partition_output = get_partition_for_one_frame(blocks, block_size, qps, frame_type)

    prob = np.zeros(num_samples)

    for i in range(num_samples):
        #print(partition_output[i])
        prob[i] = np.argmax(partition_output[i])
    
    prob = prob.astype(int)
    
    print(prob)
    #prob_arr = np.reshape(prob.astype(str), [1, (valid_height // block_size) * (valid_width // block_size) )])
    
    prob_str= "".join(prob.astype(str))
    
    with open(save_file, 'a') as f_out:
        f_out.write(prob_str)



assert len(sys.argv) == 7
yuv_file = sys.argv[1]
width = int(sys.argv[2])
height = int(sys.argv[3])
qp_one_frame = int(sys.argv[4])
frame_type = int(sys.argv[5])
frame_number = int(sys.argv[6])



t1 = time.time()
#prob_64(yuv_file, SAVE_FILE, qp_one_frame, width, height, frame_number, frame_type)
#prob_32(yuv_file, SAVE_FILE, qp_one_frame, width, height, frame_number, frame_type)
prob_16(yuv_file, SAVE_FILE, qp_one_frame, width, height, frame_number, frame_type)
t2 = time.time()
print('--------\n\nPredicting Time: %.3f sec.\n\n--------' % float(t2-t1))

