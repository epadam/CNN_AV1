## Training Process

### Data Collection

1. Replace encodeframe.c in the original encoder source file. After encoding a video frame, a txt file with partition modes of each 64x64, 32x32, 16x16 blocks is saved. The format is showing below:
ex:
0 0 9 4 8 3 120

| Frame Type           | Frame Number                | Block Size<br>(index)    | Row                   | col                  | Partition Mode                                                                                            | QP    |
|----------------------|-----------------------------|--------------------------|-----------------------|----------------------|-----------------------------------------------------------------------------------------------------------|-------|
| 0: intra<br>1: inter | Real order <br>in the video | 12: 64<br>9: 32<br>6: 16 | 1 unit = <br>4 pixels | 1 unit =<br>4 pixels | 0: NONE<br>1: Horizontal<br>2: Vertical<br>3: Split<br>4: HA<br>5: HB<br>6: VA<br>7: VB<br>8: H4<br>9: V4 | 0-255 |

2. Copy the file into DP.xlsm to seperate the partition modes of 64x64, 32x32 and 16x16 blocks. And save it as another xls file. 

3. Use rearrange_video.py to transform the video frame into block based. (Require the raw video and the xls file)

4. Repeat the steps above for extracting all the partition modes from the encoder.

5. Use merge_sample.py and generate_label_qp.py to merge all the samples and generate the full dataset.

6. Use shuffle.py to shuffle the dataset.
