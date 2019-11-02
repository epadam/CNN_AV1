import os, sys, time, math
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, backend
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt
import keras_model_64b, keras_model_64_mnist, keras_model_32b, keras_model_32_mnist, keras_model_16b, keras_model_16_mnist

num_classes = 10
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
def get_partition_for_one_frame(blocks, block_size, qps, frame_type, num_samples):

    
    if block_size == 64:
        #model = cnn_64.net() 
        if frame_type == 0:
            model=load_model('./logs/mergeaSR/64/m1/m1_qp120_64_aSR.h5')
            model2=load_model('./logs/mergeanoS/64/m2/m2_qp120_64_anoS.h5')
        else:
            model=load_model('model/inter/64')
    elif block_size == 32:
        #model = cnn_32.net() 
        if frame_type == 0:
            model=load_model('./logs/mergeaSR/32/m1/m1_qp120_32_aSR.h5')
            model2=load_model('./logs/mergeanoS/32/m2/m2_qp120_32_anoS.h5')
        else:
            model=load_model('model/inter/32')
    else:
        #model = cnn_16.net() 
        if frame_type == 0:
            model=load_model('./logs/mergeaSR/16/m1/m1_qp120_16_aSR.h5')
            model2=load_model('./logs/mergeanoS/16/m2/m2_qp120_16_anoS.h5')
        else:
            model=load_model('model/inter/16')

    result = model.predict([blocks, qps], verbose=0)
    result2 = model2.predict([blocks, qps], verbose=0)


    prob = np.zeros(num_samples)
    prob2 = np.zeros(num_samples)


    for i in range(num_samples):
        prob[i] = np.argmax(result[i])
        prob2[i] = np.argmax(result2[i])
        if prob2[i] == 3:
            prob2[i] = 4
        elif prob2[i] == 4:
            prob2[i] = 5
        elif prob2[i] == 5:
            prob2[i] = 6
        elif prob2[i] == 6:
            prob2[i] = 7
        elif prob2[i] == 7:
            prob2[i] = 8
        elif prob2[i] == 8:
            prob2[i] = 9             
        
        if prob[i] == 0:
            prob[i] = prob2[i]
        else :
            prob[i]=3
        
    #print(prob2)

    
    return prob




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
    probs = get_partition_for_one_frame(blocks, block_size, qps, frame_type, num_samples)

    
    probs = probs.astype(int)

    
    # if frame_height == 2160:
    #     probs[-60:] = 3
    # if frame_height == 1080:
    #     probs[-30:] = 3
    # elif frame_height == 720:
    #     probs[-20:] = 3
    # elif frame_height == 480:
    #     probs[-12:] = 3
    # elif frame_height == 288:
    #     probs[-6:] = 3

    # if frame_width == 352:
    #     probs[5:30:6] = 3
    # elif frame_width == 720:
    #     probs[11:96:12] =3
    

    #print(probs)
    
    prob_str= "".join(probs.astype(str))
    
    with open(save_file, 'w') as f_out:
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
    probs = get_partition_for_one_frame(blocks, block_size, qps, frame_type, num_samples)

    probs = probs.astype(int)

    
    # if frame_height == 2160:
    #     probs[-240:] = 3
    # if frame_height == 1080:
    #     probs[-60:] = 3
    # elif frame_height == 720:
    #    probs[-40:] = 3
    
    
    #print(probs)
    
    prob_str= "".join(probs.astype(str))
    
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
    probs = get_partition_for_one_frame(blocks, block_size, qps, frame_type, num_samples)    
    
    probs = probs.astype(int)


    
    # if frame_height == 1080:
    #     probs[-120:] = 3

    
    
    #print(probs)
    
    prob_str= "".join(probs.astype(str))
    
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
prob_64(yuv_file, SAVE_FILE, qp_one_frame, width, height, frame_number, frame_type)
prob_32(yuv_file, SAVE_FILE, qp_one_frame, width, height, frame_number, frame_type)
#prob_16(yuv_file, SAVE_FILE, qp_one_frame, width, height, frame_number, frame_type)
t2 = time.time()
print('--------\n\nPredicting Time: %.3f sec.\n\n--------' % float(t2-t1))

