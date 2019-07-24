# CNN_AV1
A CNN based AV1 encoder for intra frame encoding
AV1 version:


The following source files are modified:

aomenc.c

aom_codec.h

aom_codec_internal.h

av1_cx_iface.c

encoder.c

encoder.h

encode_frame.c


Process:
1. Replace it in the original AV1 source files and compile the encoder.
2. Add all the files in CNN folder to the same folder of built encoder


NOTE:
Right now it only supports 4:2:0 yuv format (y4m not support). 