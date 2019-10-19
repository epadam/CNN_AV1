# CNN_AV1
A CNN based AV1 encoder for intra frame encoding

Detailed description of the concept and implementation can be found here : https://cnn-av1-intra-encoder.readthedocs.io/en/latest/index.html

AV1 version: 1.0.0-2231-g9666276

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

### 4.  Build the encoder with cmake and make

### 5.  Include the files in CNN folder into the built encoder folder


NOTE:
1. Right now it only supports 4:2:0 yuv format (y4m not support) and resolution of 1080p and below. 
2. For resolution of 1080p and above, encoder sometimes crash. This may be because some predicted partition modes violate the rules in the encoder. This issue will be solved in the future. Also, python models will be ported into C with Tensorflow C API.
