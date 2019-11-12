import glob
import pandas as pd
import numpy as np
import os

dirpath = os.getcwd()
foldername = os.path.basename(dirpath)
k=0
for filename in sorted(glob.glob('*.xlsx')):
	print(filename)
	xlsx = pd.ExcelFile(filename)
	data64 = pd.read_excel(xlsx, sheet_name = '64', usecols="C")
	data32 = pd.read_excel(xlsx, sheet_name = '32', usecols="C")
	data16 = pd.read_excel(xlsx, sheet_name = '16', usecols="C")
	number64 = np.shape(data64)
	print(number64)
	number32 = np.shape(data32)
	print(number32)
	number16 = np.shape(data16)
	print(number16)

	name = filename.split(sep='-')
	qp_value = name[1]
	qps_64 = np.full(number64, qp_value, np.int)
	qps_32 = np.full(number32, qp_value, np.int)
	qps_16 = np.full(number16, qp_value, np.int)
	labels_64 = foldername + '_labels_64_intra.txt'
	qpfile_64 = foldername + '_qps_64_intra.txt'
	labels_32 = foldername + '_labels_32_intra.txt'
	qpfile_32 = foldername + '_qps_32_intra.txt'
	labels_16 = foldername + '_labels_16_intra.txt'
	qpfile_16 = foldername + '_qps_16_intra.txt'
	if k==0:
		l64 = open(labels_64, "w+")
		l32 = open(labels_32, "w+")
		l16 = open(labels_16, "w+")
		qp64 = open(qpfile_64, "w+")
		qp32 = open(qpfile_32, "w+")
		qp16 = open(qpfile_16, "w+")
	else:
		l64 = open(labels_64, "a+")
		l32 = open(labels_32, "a+")
		l16 = open(labels_16, "a+")
		qp64 = open(qpfile_64, "a+")
		qp32 = open(qpfile_32, "a+")
		qp16 = open(qpfile_16, "a+")

	np.savetxt(l64, data64.values, fmt = '%d')
	np.savetxt(l32, data32.values, fmt = '%d')
	np.savetxt(l16, data16.values, fmt = '%d')

	
	np.savetxt(qp64, qps_64, fmt = '%d')
	np.savetxt(qp32, qps_32, fmt = '%d')
	np.savetxt(qp16, qps_16, fmt = '%d')
	k=1



	

'''

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
'''
