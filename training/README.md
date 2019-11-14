## Training Process

### Data Collection

1. Replace encodeframe.c in the original encoder source file. After encoding a video frame, a txt file with partition modes of each 64x64, 32x32, 16x16 blocks is saved.

| Frame Type        | Frame Number             | Block Size (index) | Row                | col               | Partition Mode                                                                 | QP    |
|-------------------|--------------------------|--------------------|--------------------|-------------------|--------------------------------------------------------------------------------|-------|
| 0: intra 1: inter | Real order  in the video | 12: 64 9: 32 6: 16 | 1 unit =  4 pixels | 1 unit = 4 pixels | 0: NONE 1: Horizontal 2: Vertical 3: Split 4: HA 5: HB 6: VA 7: VB 8: H4 9: V4 | 0-255 |

2. 
