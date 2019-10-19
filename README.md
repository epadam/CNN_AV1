# CNN_AV1
A CNN based AV1 encoder for intra frame encoding

AV1 version: 1.0.0-2231-g9666276

## Usage 

#### Clone AOM AV1 from the repo:

 `git clone https://aomedia.googlesource.com/aom`

### Go to version 1.0.0-2231-g9666276:

 `git checkout 2231-g9666276`


#### Replace following files:

aom/src/aom_codec.h

apps/aomenc.c

av1/encoder/encoder.h

av1/encoder/encoder.c

av1/av1_cx_iface.c

av1/encoder/encode_frame.c

### Build the encoder with cmake and make

### Include the files in CNN folder into the built encoder folder


NOTE:
1. Right now it only supports 4:2:0 yuv format (y4m not support) and resolution of 1080p and below. 
2. For resolution of 1080p and above, encoder sometimes crash. This may be because of the violation of some predicted partition modes. This issue will be solved in the future.
