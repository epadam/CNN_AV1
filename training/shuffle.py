import numpy as np
import pandas as pd
import os, glob, sys
from sklearn.utils import shuffle
NUM_CHANNELS=1

dirpath = os.getcwd()
foldername = os.path.basename(dirpath)


block_size = 64

sample_file = foldername + '_samples_64_intra.txt'
label_file = foldername + '_labels_64_intra.txt'
qp_file = foldername + '_qps_64_intra.txt'
#read samples
f= open(sample_file, 'rb')
pixels = f.read()
data = np.frombuffer(pixels, dtype = np.float) 
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

 
print(data[0][0][0])
print(single_label[320])
print(qps[0])

data, single_label, qps = shuffle(data, single_label, qps)
 
print(data[0][0][0])
print(single_label[320])
print(qps[0])

f.close()
f_single_label.close()
f_qp.close()

shuffle_sample_file = 'sh_samples_64_intra.txt'
shuffle_label_file ='sh_labels_64_intra.txt'
shuffle_qp_file = 'sh_qps_64_intra.txt'


f= open(shuffle_sample_file, 'wb+')
f.write(data) 

f_single_label = open(shuffle_label_file, 'w+')
np.savetxt(f_single_label, single_label, fmt = '%d')

f_qp = open(shuffle_qp_file, 'w+') 
np.savetxt(f_qp, qps, fmt = '%d')

f.close()
f_single_label.close()
f_qp.close()

'''

block_size = 32
sample_file = foldername + '_samples_32_intra.txt'
label_file = foldername + '_labels_32_intra.txt'
qp_file = foldername + '_qps_32_intra.txt'

#read samples
f= open(sample_file, 'rb')
pixels = f.read()
data = np.frombuffer(pixels, dtype = np.float)  
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

 
print(data[0][0][0])
print(single_label[320])
print(qps[0])

data, single_label, qps = shuffle(data, single_label, qps)
 
print(data[0][0][0])
print(single_label[320])
print(qps[0])

f.close()
f_single_label.close()
f_qp.close()



shuffle_sample_file = 'sh_samples_32_intra.txt'
shuffle_label_file ='sh_labels_32_intra.txt'
shuffle_qp_file = 'sh_qps_32_intra.txt'


f= open(shuffle_sample_file, 'wb+')
f.write(data) 

f_single_label = open(shuffle_label_file, 'w+')
np.savetxt(f_single_label, single_label, fmt = '%d')

f_qp = open(shuffle_qp_file, 'w+') 
np.savetxt(f_qp, qps, fmt = '%d')

f.close()
f_single_label.close()
f_qp.close()
'''

'''

block_size = 16
sample_file = foldername + '_samples_16_intra.txt'
label_file = foldername + '_labels_16_intra.txt'
qp_file = foldername + '_qps_16_intra.txt'

#read samples
f= open(sample_file, 'rb')
pixels = f.read()
data = np.frombuffer(pixels, dtype = np.float)
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

 
print(data[0][0][0])
print(single_label[320])
print(qps[0])

data, single_label, qps = shuffle(data, single_label, qps)
 
print(data[0][0][0])
print(single_label[320])
print(qps[0])


f.close()
f_single_label.close()
f_qp.close()

shuffle_sample_file = 'sh_samples_16_intra.txt'
shuffle_label_file ='sh_labels_16_intra.txt'
shuffle_qp_file = 'sh_qps_16_intra.txt'


f= open(shuffle_sample_file, 'wb+')
f.write(data) 

f_single_label = open(shuffle_label_file, 'w+')
np.savetxt(f_single_label, single_label, fmt = '%d')

f_qp = open(shuffle_qp_file, 'w+') 
np.savetxt(f_qp, qps, fmt = '%d')
'''


'''
assert len(sys.argv) == 2
label_excel = sys.argv[1]
#frame_number = int(sys.argv[2])


file = os.path.splitext(label_excel)[0]
name = file.split(sep='-')
yuv_name = name[0]+'.yuv'
print(yuv_name)
qp = name[1]
frame_number = int(name[2])
mode = name[3]
image_size = 64

#image_size_str =str(image_size)
save_name = "".join(file+'_raw_'+image_size+'.txt')
print(save_name)


for filename in glob.glob('*_16.txt'):
	print(filename)


cols =10
index =0

samples = np.arange(10)
labels =np.array([0,1,4,7,8])
no_labels = labels.size
batch_size = 10

for col in range(cols):
	print(col)
	if labels[index] != col:
	    samples = np.delete(samples, index)
	elif index == no_labels-1:
		break
	else:
	    index +=1
	print(samples)

samples = samples[0:no_labels]
print(samples)



valid_height = 320
valid_width = 384
image_size = 64

rows = valid_height // image_size
cols = valid_width // image_size
col_arr = np.arange(cols)

filename = '160-bus_cif_intra.xlsx'
xlsx = pd.ExcelFile(filename)
lcols = pd.read_excel(xlsx, sheet_name = '64', usecols="B")
#label_cols = pd.read_excel(xlsx, sheet_name = '64', usecols="B")
lcols = lcols.values/16
lcols = lcols.astype(int)
num_label = lcols.size
#print(lcols)

index = 0
for row in range(rows):
	for col in range(cols):
		print(index, col, lcols[index])
		if lcols[index] != col:
			print(index, col, lcols[index])
			#samples = np.delete(samples, index)
		elif index == num_label-1:
			break
		else:
			index +=1





label_rows = label_rows/16
label_rows = label_rows.astype(int)



label_cols = label_cols.values
label_cols = label_cols/16
label_cols = label_cols.astype(int)

valid_height = 320
valid_width = 384
image_size = 64

rows = valid_height // image_size
cols = valid_width // image_size
col_arr = np.arange(cols)







import glob
import pandas as pd
import numpy as np
import os



for filename in glob.glob('*.xlsx'):
	print(filename)
	xlsx = pd.ExcelFile(filename)
	data64 = pd.read_excel(xlsx, sheet_name = '64', usecols="C")
	data32 = pd.read_excel(xlsx, sheet_name = '32', usecols="C")
	data16 = pd.read_excel(xlsx, sheet_name = '16', usecols="C")
	number64 = np.shape(data64)
	print(number64)
	number32 = np.shape(data32)
	number16 = np.shape(data16)

	name = filename.split(sep='-')
	qp_value = name[0]
	qps_64 = np.full(number64, qp_value, np.int)
	qps_32 = np.full(number32, qp_value, np.int)
	qps_16 = np.full(number16, qp_value, np.int)
	labels_64 = 'labels_64_intra.txt'
	qpfile_64 = 'qps_64_intra.txt'
	labels_32 = 'labels_32_intra.txt'
	qpfile_32 = 'qps_32_intra.txt'
	labels_16 = 'labels_16_intra.txt'
	qpfile_16 = 'qps_16_intra.txt'

	l64 = open(labels_64, "a+")
	l32 = open(labels_32, "a+")
	l16 = open(labels_16, "a+")
	np.savetxt(l64, data64.values, fmt = '%d')
	np.savetxt(l32, data32.values, fmt = '%d')
	np.savetxt(l16, data16.values, fmt = '%d')

	qp64 = open(qpfile_64, "a+")
	qp32 = open(qpfile_32, "a+")
	qp16 = open(qpfile_16, "a+")
	np.savetxt(qp64, qps_64, fmt = '%d')
	np.savetxt(qp32, qps_32, fmt = '%d')
	np.savetxt(qp16, qps_16, fmt = '%d')




	



close()
	close(labels_32)
	close(labels_16)
	close(qpsave_64)
	close(qpsave_32)
	close(qpsave_16)

filename = '160-park_joy_420_720p50.xlsx'

xlsx = pd.ExcelFile(filename)
data64 = pd.read_excel(xlsx, sheet_name = '64', usecols="C")
number64 = np.shape(data64)
print(number64)

name = filename.split(sep='-')
qp_value = name[0]
qps_64 = np.full(number64, qp_value)
print(type(qps_64[0][0]))

qpsave_64 = 'qps_64_intra.txt'
qp64 = open(qpsave_64, 'a+')

np.savetxt(qp64, qps_64, fmt = '%d')



f_single_label = open('test.txt', 'r')
single_label = f_single_label.read()    
single_label =np.fromstring (single_label, dtype=np.int32 ,sep=' ')
print(np.shape(single_label))
#single_label = np.reshape(single_label, [-1]) 

#xls.sheet_names # see all sheet names
#data.get('64') # get a specific sheet to DataFrame

filename = 'park_joy_420_720p50-160.xlsx'

name = filename.split(sep='.')
suffix = '.txt'
filename64 = "".join(name[0]+'_intra'+suffix)

print(filename64)




x = np.array([33,19,82,11,37])  

x = np.delete(x, np.s_[2:3])
print(x)
#x = np.delete(x, [0,1])
print(x)
'''
