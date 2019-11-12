import numpy as np
import math, sys, glob, os
import pandas as pd

dirpath = os.getcwd()
foldername = os.path.basename(dirpath)
suffix = '.txt'
image_size = 16

image_size_str=str(image_size)
name = "".join('_'+image_size_str+'_intra'+suffix)

smp = "".join('new_samples'+name)
lbn = "".join('new_labels'+name)
qpn = "".join('new_qps'+name)


training_samples_all = 'all_samples_'+image_size_str+suffix
training_labels_all = 'all_labels_'+image_size_str+suffix
training_qps_all = 'all_qps_'+image_size_str+suffix


f_samples = open (smp, 'rb')
cdata = f_samples.read()
training_data = np.frombuffer(cdata, dtype=float)
print(np.shape(training_data))

f_labels = open (lbn, 'r')
labels = f_labels.read()
#labels =np.fromstring (labels, dtype=np.float, sep=' ')

f_qps = open (qpn, 'r')
qps = f_qps.read()
#qps =np.fromstring (qps, dtype=np.float, sep=' ')



f_training = open(training_samples_all, 'ab+')
f_l = open(training_labels_all, 'a+')
f_q = open(training_qps_all, 'a+')


f_training.write(training_data)
f_l.write(labels)
f_q.write(qps)



f_check = open (training_samples_all, 'rb')
cdata = f_check.read()
training_data = np.frombuffer(cdata, dtype=float)
print(np.shape(training_data))
training_data = np.reshape(training_data, [-1, image_size, image_size, 1])
print(np.shape(training_data))

l_check = open (training_labels_all, 'r')
ls = l_check.read()
ls =np.fromstring (ls, dtype=np.float, sep=' ')
print(np.shape(ls))

q_check = open (training_qps_all, 'r')
qs = q_check.read()
qs =np.fromstring (qs, dtype=np.float, sep=' ')
print(np.shape(qs))



'''
	k=0
	for filename in sorted(glob.glob(smp)):
		lbn = "".join('_labels_'+name)
		qpn = "".join('_qps_'+name)

		f_samples = open (filename, 'rb')
		cdata = f_samples.read()
		training_data = np.frombuffer(cdata, dtype=float)
		print(np.shape(training_data))

		f_labels = open (lbn, 'r')
		labels = f_labels.read()
		labels =np.fromstring (labels, dtype=np.float, sep=' ')


		f_qps = open (qpn, 'r')
		qps = f_qps.read()
		qps =np.fromstring (qps, dtype=np.float, sep=' ')

		

		if k==0:
			f_training = open(training_samples_all, 'wb+')
			f_l = open(training_labels_all, 'w+')
			f_q = open(training_qps_all, 'w+')

		else:
			f_training = open(training_sample_all, 'ab+')
			f_l = open(training_labels_all, 'a+')
			f_q = open(training_qps_all, 'a+')
		
		f_training.write(training_data)
		f_l.write(labels)
		f_q.write(qps)


		k =1


image_size = image_size//2
'''

	

	