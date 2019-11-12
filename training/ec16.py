import os, sys, math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

block_size = 16
NUM_CHANNELS=1

dirpath = os.getcwd()
foldername = os.path.basename(dirpath)
sample_file = 'ash_samples_16_intra.txt'
label_file = 'ash_labels_16_intra.txt'
qp_file = 'ash_qps_16_intra.txt'
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

#read qps
f_qp = open(qp_file, 'r') 
qps = f_qp.read()
qps =np.fromstring (qps, dtype=np.float, sep=' ')
qps = np.reshape(qps, [-1,1]) 
print(np.shape(qps)) 

f.close()
f_single_label.close()
f_qp.close()

label_None = np.zeros(700000)
label_H = np.zeros(110000)
label_V = np.zeros(110000)
label_S = np.zeros(100000)
label_HA = np.zeros(20000)
label_HB = np.zeros(20000)
label_VA = np.zeros(20000)
label_VB = np.zeros(20000)
label_H4 = np.zeros(80000)
label_V4 = np.zeros(40000)



qp_None = np.zeros((700000,1))
qp_H = np.zeros((110000,1))
qp_V = np.zeros((110000,1))
qp_S = np.zeros((100000,1))
qp_HA = np.zeros((20000,1))
qp_HB = np.zeros((20000,1))
qp_VA = np.zeros((20000,1))
qp_VB = np.zeros((20000,1))
qp_H4 = np.zeros((80000,1))
qp_V4 = np.zeros((40000,1))

data_None = np.zeros([700000,block_size, block_size, NUM_CHANNELS])
data_H = np.zeros([110000,block_size, block_size, NUM_CHANNELS])
data_V = np.zeros((110000,block_size, block_size, NUM_CHANNELS))
data_S = np.zeros((100000,block_size, block_size, NUM_CHANNELS))
data_HA = np.zeros((20000,block_size, block_size, NUM_CHANNELS))
data_HB = np.zeros((20000,block_size, block_size, NUM_CHANNELS))
data_VA = np.zeros((20000,block_size, block_size, NUM_CHANNELS))
data_VB = np.zeros((20000,block_size, block_size, NUM_CHANNELS))
data_H4 = np.zeros((80000,block_size, block_size, NUM_CHANNELS))
data_V4 = np.zeros((40000,block_size, block_size, NUM_CHANNELS))

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
		label_None[index0] = single_label[index]
		qp_None[index0] = qps[index]
		data_None[index0] = data[index]
		index0 +=1
	elif single_label[index] == 1:
		label_H [index1]= single_label[index]
		qp_H [index1]= qps[index]
		data_H [index1]= data[index]
		index1 +=1
	elif single_label[index] == 2:
		label_V [index2]= single_label[index]
		qp_V [index2]= qps[index]
		data_V [index2]= data[index]
		index2 +=1
	elif single_label[index] == 3:
		label_S [index3]= single_label[index]
		qp_S [index3]= qps[index]
		data_S [index3]= data[index]
		index3 +=1
	elif single_label[index] == 4:
		label_HA [index4]= single_label[index]
		qp_HA [index4]= qps[index]
		data_HA [index4]= data[index]
		index4 +=1
	elif single_label[index] == 5:
		label_HB [index5]= single_label[index]
		qp_HB [index5]= qps[index]
		data_HB [index5]= data[index]
		index5 +=1
	elif single_label[index] == 6:
		label_VA [index6]= single_label[index]
		qp_VA [index6]=  qps[index]
		data_VA [index6]= data[index]
		index6 +=1
	elif single_label[index] == 7:
		label_VB [index7]= single_label[index]
		qp_VB [index7]= qps[index]
		data_VB [index7]= data[index]
		index7 +=1
	elif single_label[index] == 8:
		label_H4 [index8]= single_label[index]
		qp_H4 [index8]= qps[index]
		data_H4 [index8]= data[index]
		index8 +=1
	elif single_label[index] == 9:
		label_V4 [index9]= single_label[index]
		qp_V4 [index9]= qps[index]
		data_V4 [index9]= data[index]
		index9 +=1

'''
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

'''

print(index0, index1, index2, index3, index4, index5, index6, index7, index8, index9)

K = np.array([index0, index1, index2, index3, index4, index5, index6, index7, index8, index9] )

total = sum(K)

perc = np.array([round((index0/total*100),2), round((index1/total*100),2), round((index2/total*100),2), round((index3/total*100),2),round((index4/total*100),2),round((index5/total*100),2),round((index6/total*100),2),round((index7/total*100),2),round((index8/total*100),2),round((index9/total*100),2)])

Index = np.arange(len(K))

label = ['None', 'Hor', 'Ver', 'Split', 'Hor_A', 'Hor_B', 'Ver_A', 'Ver_B', 'Hor_4', 'Ver_4']

plt.bar(Index, perc, align='center', alpha = 0.5)
plt.ylabel('percentage (%)',fontsize=10)
plt.xlabel('partition mode',fontsize=10)
plt.xticks(Index, label, fontsize=10)
plt.ylim(top=100)
plt.title('11 frames, block size=16, qp=120, total smaples='+str(total), fontsize=10)
for i in range(len(K)):
    plt.text(x=Index[i]-0.2, y=perc[i]+0.7, s = perc[i], fontsize=8)
plt.savefig(foldername +'_distribution_16.jpg')
plt.show



NUM_BATCH = np.amin(K)

final_data = np.concatenate((data_None[0:NUM_BATCH], data_H[0:NUM_BATCH], data_V[0:NUM_BATCH], data_HA[0:NUM_BATCH], data_HB[0:NUM_BATCH], data_VA[0:NUM_BATCH],data_VB[0:NUM_BATCH], data_H4[0:NUM_BATCH],data_V4[0:NUM_BATCH]))
print(np.shape(final_data))
final_label =np.concatenate((label_None[0:NUM_BATCH], label_H[0:NUM_BATCH], label_V[0:NUM_BATCH],  label_HA[0:NUM_BATCH], label_HB[0:NUM_BATCH],label_VA[0:NUM_BATCH],label_VB[0:NUM_BATCH],label_H4[0:NUM_BATCH],label_V4[0:NUM_BATCH]))
print(final_label)
final_qp = np.concatenate((qp_None[0:NUM_BATCH], qp_H[0:NUM_BATCH], qp_V[0:NUM_BATCH], qp_HA[0:NUM_BATCH], qp_HB[0:NUM_BATCH],qp_VA[0:NUM_BATCH],qp_VB[0:NUM_BATCH],qp_H4[0:NUM_BATCH],qp_V4[0:NUM_BATCH]))
print(np.shape(final_qp))


final_data, final_label, final_qp = shuffle(final_data, final_label, final_qp)

print(np.shape(final_data))
print(np.shape(final_label))
print(np.shape(final_qp))

shuffle_sample_file = foldername + 'anoS_samples_16_intra.txt'
shuffle_label_file = foldername + 'anoS_labels_16_intra.txt'
shuffle_qp_file = foldername + 'anoS_qps_16_intra.txt'


f= open(shuffle_sample_file, 'wb+')
f.write(final_data) 

f_single_label = open(shuffle_label_file, 'w+')
np.savetxt(f_single_label, final_label, fmt = '%d')

f_qp = open(shuffle_qp_file, 'w+') 
np.savetxt(f_qp, final_qp, fmt = '%d')




'''

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
'''

#cdata = cdata.reshape([36, 64, 64])
#plt.imshow(cdata[2])
#plt.show() 

'''
f_single_label = open('bus_cif-160.txt', 'r')
single_label = f_single_label.read()    
single_label =np.fromstring (single_label, dtype=np.int32 ,sep=' ')
print(np.shape(single_label))
'''
