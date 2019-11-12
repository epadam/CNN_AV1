import numpy as np
import math, sys, os
import pandas as pd

assert len(sys.argv) == 4
label_excel = sys.argv[1]
frame_width = int(sys.argv[2])
frame_height = int(sys.argv[3])

file = os.path.splitext(label_excel)[0]
name = file.split(sep='-')
yuv_file = name[0]+'.yuv'
#yuv_file = '720p50_shields_ter_f3.yuv'
frame_number = int(name[2])
#frame_number =0





#open yuv file and read the required frame into array
f_video= open(yuv_file, 'rb')
f_video.seek(frame_number*(frame_width * frame_height + frame_width * frame_height//2), 0) # find the frame 
y_buf = f_video.read(frame_width * frame_height)
data = np.frombuffer(y_buf, dtype = np.uint8)
data = data.reshape(frame_height, frame_width)

image_size = 64
for i in range (3):	
	image_size_str =str(image_size)
	raw_sample_file = "".join(file+'_raw_'+image_size_str+'.txt')

	# extend the frame size to multiple of image_size
	rows = math.ceil(frame_height / image_size)
	cols = math.ceil(frame_width / image_size)
	print(rows, cols)
	valid_height = rows * image_size
	valid_width = cols * image_size
	if valid_height > frame_height:
	    data = np.concatenate((data, np.zeros((valid_height - frame_height, frame_width))), axis = 0)
	if valid_width > frame_width:
		data = np.concatenate((data, np.zeros((valid_height, valid_width - frame_width))), axis = 1)
	

	#rearrange data into block based
	batch_size = rows * cols
	print(batch_size)
	input_batch = np.zeros((batch_size, image_size, image_size))

	index = 0
	ystart = 0
	while ystart < valid_height:
		xstart = 0
		while xstart < valid_width:
			input_batch[index] = data[ystart : ystart + image_size, xstart : xstart + image_size]
			index += 1
			xstart += image_size
		ystart += image_size

	#--------------------------------------------------------------------------
	#Delete the blocks without labels
	xlsx = pd.ExcelFile(label_excel)
	sheet = str(image_size)
	lcols = pd.read_excel(xlsx, sheet_name = sheet, usecols="B")

	lcols = (lcols.values/image_size)*4
	lcols = lcols.astype(int)
	num_label = lcols.size

	index = 0
	for row in range(rows):
		for col in range(cols):
			if lcols[index] != col:
				input_batch = np.delete(input_batch, index, axis=0)
				print(index, col, lcols[index])
			elif index == num_label-1:
				break
			else:
				index +=1

	input_batch = input_batch[0:num_label]
	print(np.shape(input_batch))
	#convert to 1D array
	input_batch=input_batch.flatten()


	#write into txt file
	f_training = open(raw_sample_file, 'wb+') 
	f_training.write(input_batch) 
	f_video.close()
	f_training.close()


	#check if number of bytes is correct
	f_check = open (raw_sample_file, 'rb')
	cdata = f_check.read()
	#print(type(training_data[0]))	
	training_data = np.frombuffer(cdata, dtype=float)
	print(np.shape(training_data))

	image_size=image_size//2






#Abandoned
'''
index=0
iiy=0
iix=0
iy=0
ix=0
while iiy < valid_height:
	while iix < valid_width:
		iyy = iiy		
		while iy < 2:
			ixx = iix
			while ix < 2:
				print(iy, ix, iiy, iix, index)
				input_batch[index]= data[iiy:iiy+image_size, iix:iix+image_size]
				index +=1
				ix+=1
				iix += image_size
			iy+=1
			iiy+=image_size
			ix=0
			iix=ixx
		iy=0
		iiy=iyy
		iix += 2*image_size	
	iiy += 2*image_size
	iix = 0
'''


'''
f_flat = open("flatten.txt", 'rb') 

#flat_write = f_flat.write(input_batch)

flat_read = f_flat.read()
reading = np.frombuffer(flat_read, dtype=float)
print(reading[0], reading[33])
'''
