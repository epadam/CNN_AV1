# CNN AV1 Intra Encoder
A CNN based AV1 intra encoder. This encoder uses CNN for 10 partition modes prediction for AV1 intra encoding.

Detailed description of the concept and implementation can be found here : https://cnn-av1-intra-encoder.readthedocs.io/en/latest/index.html

AV1 version: 1.0.0-2231-g9666276

## Performance 

Encoding time comparison (1080p and 720p)

QP=120, 1 thread (prediction for 64x64, 32x32, 16x16 blocks)
![performance](https://cnn-av1-intra-encoder.readthedocs.io/en/latest/_images/encoding_time_cnn.png)

Encoding time comparison (4K)

QP=120, 1 threads, (prediction for 64x64, 32x32, 16x16 disabled)
![performance4K](https://cnn-av1-intra-encoder.readthedocs.io/en/latest/_images/EncodingTimeComparison.png)

PSNR and file size comparison (4K)
![performance4K_2](https://cnn-av1-intra-encoder.readthedocs.io/en/latest/_images/PSNR(dB)andBits(4K).png)



## Usage 

### Prerequisites

Tensorflow

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
1. Right now it only supports 4:2:0 yuv format and the models are only trained with QP=120 at the moment.  
2. Predictions for 16x16 blocks are disabled to avoid encoding fail temporally. (This may be because some predicted partition modes cause overflow in the transform step. This issue will be solved in the future.
