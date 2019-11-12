import os, sys, math
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt




NUM_BATCH = 20
block_size = 64
NUM_CHANNELS=1

dirpath = os.getcwd()
foldername = os.path.basename(dirpath)

sample_file = 'HS_samples_64_intra.txt'
label_file = 'HS_labels_64_intra.txt'
qp_file = 'HS_qps_64_intra.txt'
#read samples
f= open(sample_file, 'rb')
pixels = f.read()
data = np.frombuffer(pixels, dtype = np.float)
#data = np.reshape(data, [28, 64*64*NUM_CHANNELS])  
data = np.reshape(data, [-1, block_size, block_size])
print(np.shape(data))  
  
       
#read labels
f_single_label = open(label_file, 'r')
single_label = f_single_label.read()    
single_label =np.fromstring (single_label, dtype=np.int32 ,sep=' ')
single_label = np.reshape(single_label, [-1]) 
print(np.shape(single_label))  

#read qps
f_qp = open(qp_file, 'r') 
qps = f_qp.read()
qps =np.fromstring (qps, dtype=np.float, sep=' ')
qps = np.reshape(qps, [-1,1]) 
print(np.shape(qps)) 



for i in range(np.size(qps)):
	print(single_label[i], qps[i,0])
	plt.imshow(data[i], cmap='gray')
	plt.colorbar()
	plt.show()




'''
0 label_None = np.zeros(0) 
1 label_H = np.zeros(0)
2 label_V = np.zeros(0)
3 label_S = np.zeros(0)
4 label_HA = np.zeros(0)
5 label_HB = np.zeros(0)
6 label_VA = np.zeros(0)
7 label_VB = np.zeros(0)
8 label_H4 = np.zeros(0)
9 label_V4 = np.zeros(0)



qp_None = np.zeros(0)
qp_H = np.zeros(0)
qp_V = np.zeros(0)
qp_S = np.zeros(0)
qp_HA = np.zeros(0)
qp_HB = np.zeros(0)
qp_VA = np.zeros(0)
qp_VB = np.zeros(0)
qp_H4 = np.zeros(0)
qp_V4 = np.zeros(0)

data_None = np.zeros(0)
data_H = np.zeros(0)
data_V = np.zeros(0)
data_S = np.zeros(0)
data_HA = np.zeros(0)
data_HB = np.zeros(0)
data_VA = np.zeros(0)
data_VB = np.zeros(0)
data_H4 = np.zeros(0)
data_V4 = np.zeros(0)

index=0
index0=0
index1=0
index2=0
index3=0
index4=0
index5=0
index6=0
index7=0
index8=0
index9=0

for index in range(np.size(single_label)):
	if single_label[index] == 0:
		label_None = np.append(label_None, single_label[index])
		qp_None = np.append(qp_None, qps[index])
		data_None = np.append(data_None, data[index])
	elif single_label[index] == 1:
		label_H = np.append(label_H, single_label[index])
		qp_H = np.append(qp_H, qps[index])
		data_H = np.append(data_H, data[index])
	elif single_label[index] == 2:
		label_V = np.append(label_V, single_label[index])
		qp_V = np.append(qp_V, qps[index])
		data_V = np.append(data_V, data[index])
	elif single_label[index] == 3:
		label_S = np.append(label_S, single_label[index])
		qp_S = np.append(qp_S, qps[index])
		data_S = np.append(data_S, data[index])
	elif single_label[index] == 4:
		label_HA = np.append(label_HA, single_label[index])
		qp_HA = np.append(qp_HA, qps[index])
		data_HA = np.append(data_HA, data[index])
	elif single_label[index] == 5:
		label_HB = np.append(label_HB, single_label[index])
		qp_HB = np.append(qp_HB, qps[index])
		data_HB = np.append(data_HB, data[index])
	elif single_label[index] == 6:
		label_VA = np.append(label_VA, single_label[index])
		qp_VA = np.append(qp_VA, qps[index])
		data_VA = np.append(data_VA, data[index])
	elif single_label[index] == 7:
		label_VB = np.append(label_VB, single_label[index])
		qp_VB = np.append(qp_VB, qps[index])
		data_VB = np.append(data_VB, data[index])
	elif single_label[index] == 8:
		label_H4 = np.append(label_H4, single_label[index])
		qp_H4 = np.append(qp_H4, qps[index])
		data_H4 = np.append(data_H4, data[index])
	elif single_label[index] == 9:
		label_V4 = np.append(label_V4, single_label[index])
		qp_V4 = np.append(qp_V4, qps[index])
		data_V4 = np.append(data_V4, data[index])

data_None = np.reshape(data_None,[-1, block_size, block_size, NUM_CHANNELS])
data_H = np.reshape(data_H,[-1, block_size, block_size, NUM_CHANNELS])
data_V = np.reshape(data_V,[-1, block_size, block_size, NUM_CHANNELS])
data_S = np.reshape(data_S,[-1, block_size, block_size, NUM_CHANNELS])
data_HA = np.reshape(data_HA,[-1, block_size, block_size, NUM_CHANNELS])
data_HB = np.reshape(data_HB,[-1, block_size, block_size, NUM_CHANNELS])
data_VA = np.reshape(data_VA,[-1, block_size, block_size, NUM_CHANNELS])
data_VB = np.reshape(data_VB,[-1, block_size, block_size, NUM_CHANNELS])
data_H4 = np.reshape(data_H4,[-1, block_size, block_size, NUM_CHANNELS])
data_V4 = np.reshape(data_V4,[-1, block_size, block_size, NUM_CHANNELS])


print(np.shape(label_None))
print(np.shape(label_H))
print(np.shape(label_V))
print(np.shape(label_S))
print(np.shape(label_HA))
print(np.shape(label_HB))
print(np.shape(label_VA))
print(np.shape(label_VB))
print(np.shape(label_H4))
print(np.shape(label_V4))


final_data = np.concatenate((data_None[0:NUM_BATCH], data_H[0:NUM_BATCH], data_V[0:NUM_BATCH], data_S[0:NUM_BATCH], data_HA[0:NUM_BATCH], data_HB[0:NUM_BATCH], data_VA[0:NUM_BATCH],data_VB[0:NUM_BATCH], data_H4[0:NUM_BATCH],data_V4[0:NUM_BATCH]))
print(np.shape(final_data))
final_label =np.concatenate((label_None[0:NUM_BATCH], label_H[0:NUM_BATCH], label_V[0:NUM_BATCH], label_S[0:NUM_BATCH], label_HA[0:NUM_BATCH], label_HB[0:NUM_BATCH],label_VA[0:NUM_BATCH],label_VB[0:NUM_BATCH],label_H4[0:NUM_BATCH],label_V4[0:NUM_BATCH]))
print(final_label)
final_qp = np.concatenate((qp_None[0:NUM_BATCH], qp_H[0:NUM_BATCH], qp_V[0:NUM_BATCH], qp_S[0:NUM_BATCH], qp_HA[0:NUM_BATCH], qp_HB[0:NUM_BATCH],qp_VA[0:NUM_BATCH],qp_VB[0:NUM_BATCH],qp_H4[0:NUM_BATCH],qp_V4[0:NUM_BATCH]))
print(final_qp)

final_data, final_label, final_qp = shuffle(final_data, final_label, final_qp)



image_size = 64
frame_height =288
frame_width =352
NUM_CHANNELS =1
f = open('bridge_far_cif_frames0.yuv', 'rb')

#f.seek(frame_number*(frame_width * frame_height + frame_width * frame_height//2), 0)
y_buf = f.read(frame_width * frame_height)
data = np.frombuffer(y_buf, dtype = np.uint8)
data = data.reshape(frame_height, frame_width)

valid_height = math.ceil(frame_height / image_size) * image_size
valid_width = math.ceil(frame_width / image_size) * image_size
if valid_height > frame_height:
	data = np.concatenate((data, np.zeros((valid_height - frame_height, frame_width))), axis = 0)
if valid_width > frame_width:
	data = np.concatenate((data, np.zeros((valid_height, valid_width - frame_width))), axis = 1)

batch_size = math.ceil(frame_height / image_size) * math.ceil(frame_width / image_size)
input_batch = np.zeros((batch_size, image_size, image_size))


index = 0
ystart = 0
while ystart < frame_height:
    xstart = 0
    while xstart < frame_width:
        CU_input = data[ystart : ystart + image_size, xstart : xstart + image_size]
        input_batch[index] = np.reshape(CU_input, [1, image_size, image_size])
        
        #input_batch[index] = np.subtract(input_batch[index], np.mean(input_batch[index]))
        plt.imshow(input_batch[index])
        plt.colorbar()
        plt.show()
        index += 1
        xstart += image_size
    ystart += image_size






a = np.array([2.1, 10.5, 3.3])

a = a.astype(str)
    

print(a)

super_block_size = 128


f_video= open("block_samples.txt", 'rb')

#f_video.seek(frame_number*(frame_width * frame_height + frame_width * frame_height//2), 0)


new_buf = f_video.read()
cdata = np.frombuffer(new_buf, dtype = np.float64)
print(np.shape(cdata))

cdata = cdata.reshape([840, 64, 64])
print(cdata[839])
input_batch = np.delete(cdata, [810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839], 0) #need to delete files

input_batch=input_batch.flatten()

f_write= open("block_samples.txt", 'ab+')

f_write.write(input_batch) 


#cdata = cdata.reshape([36, 64, 64])
#plt.imshow(cdata[2])
#plt.show() 


f_single_label = open('bus_cif-160.txt', 'r')
single_label = f_single_label.read()    
single_label =np.fromstring (single_label, dtype=np.int32 ,sep=' ')
print(np.shape(single_label))
'''
