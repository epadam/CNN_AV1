# CNN AV1 Intra Encoder
A CNN based AV1 intra encoder

Detailed description of the concept and implementation can be found here : https://cnn-av1-intra-encoder.readthedocs.io/en/latest/index.html

AV1 version: 1.0.0-2231-g9666276

## Performance 

![performance](https://cnn-av1-intra-encoder.readthedocs.io/en/latest/_images/encoding_time_cnn.png)


## Usage 

### 1.  Clone AOM AV1 from the repo:

 `git clone https://aomedia.googlesource.com/aom`

### 2.  Go to version 1.0.0-2231-g9666276:

 `git checkout 2231-g9666276`


### 3.  Replace following files:

aom/src/aom_codec.h

apps/aomenc.c

av1/encoder/encoder.h

av1/encoder/encoder.c

av1/av1_cx_iface.c

av1/encoder/encode_frame.c

### 4. Follow the instruction to compile the encoder

https://aomedia.googlesource.com/aom/

### 5.  Copy the files in CNN folder into the built encoder folder


NOTE:
1. Right now it only supports 4:2:0 yuv format (y4m not support) and resolution of 1080p and below. 
2. For resolution of 1080p and above, encoder sometimes crash. This may be because some predicted partition modes violate the rules in the encoder. This issue will be solved in the future. Also, python models will be ported into C with Tensorflow C API. (For this resason, predictions for 16x16 blocks are not used to avoid encoding fail)
