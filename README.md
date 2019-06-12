# CNN_AV1
a CNN based AV1 encoder for intra frame encoding
AV1 version:

Process:
the following source files are modified, 

aomenc.c

encode_frame.c

aom_codec.h

aom_codec_internal.h

1. Replace it in the original AV1 source files and compile the encoder.
2. Add the python files in CNN folder to the same folder of built encoder


NOTE:
Right now it only supports 4:2:0 yuv format (y4m not support). 
