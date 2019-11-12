import numpy as np
import math, sys, glob, os
import pandas as pd

dirpath = os.getcwd()
foldername = os.path.basename(dirpath)
suffix = '.txt'
image_size = 64
for i in range(3):
	image_size_str=str(image_size)
	name = "".join('*intra_raw_'+image_size_str+suffix)
	training_sample_all = "".join(foldername+'_sample_'+image_size_str+suffix)
	k=0
	for filename in sorted(glob.glob(name)):
		print(filename)
		f_samples = open (filename, 'rb')
		cdata = f_samples.read()
		training_data = np.frombuffer(cdata, dtype=float)
		print(np.shape(training_data))

		if k==0:
			f_training = open(training_sample_all, 'wb+')
		else:
			f_training = open(training_sample_all, 'ab+')
		f_training.write(training_data)
		k =1

	f_training.close()

	f_check = open (training_sample_all, 'rb')
	cdata = f_check.read()
	training_data = np.frombuffer(cdata, dtype=float)
	print(np.shape(training_data))
	training_data = np.reshape(training_data, [-1, image_size, image_size, 1])
	print(np.shape(training_data))
	image_size = image_size//2

'''
image_size = 64
for i in range(3):
	image_size_str=str(image_size)
	name = "".join('*inter_raw_'+image_size_str+suffix)
	training_sample_all = "".join('training_samples_all_inter_'+image_size_str+suffix)
	k=0
	for filename in glob.glob(name):
		print(filename)
		f_samples = open (filename, 'rb')
		cdata = f_samples.read()
		training_data = np.frombuffer(cdata, dtype=float)
		print(np.shape(training_data))

		if k==0:
			f_training = open(training_sample_all, 'wb+')
		else:
			f_training = open(training_sample_all, 'ab+')
		f_training.write(training_data)
		k =1

	f_training.close()

	f_check = open (training_sample_all, 'rb')
	cdata = f_check.read()
	training_data = np.frombuffer(cdata, dtype=float)
	print(np.shape(training_data))
	training_data = np.reshape(training_data, [-1, image_size, image_size, 1])
	print(np.shape(training_data))
	image_size = image_size//2
'''
