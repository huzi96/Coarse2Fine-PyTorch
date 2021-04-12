# Coarse-to-Fine Hyper-Prior Modeling for Learned Image Compressionn
## Update
We add an experimental training code to help developers train their own compression models for a specific range of bit-rate. Please check the ```train/``` sub-directory.
## Overview
This is the implementation of ther paper,
> Yueyu Hu, Wenhan Yang, Jiaying Liu, 
> Coarse-to-Fine Hyper-Prior Modeling for Learned Image Compression,
> <i>AAAI Conference on Artificial Intelligence</i> (<i>AAAI</i>), 2020
 
and also the journal version,

> Yueyu Hu, Wenhan Yang, Zhan Ma, Jiaying Liu,
> Learning End-to-End Lossy Image Compression: A Benchmark,
> <i>IEEE Transactions on Pattern Analysis and Machine Intelligence</i> (<i>TPAMI</i>), 2021.

This is the PyTorch version of Coarse-to-Fine Hyper-Prior Model. The code load and convert weights trained with TensorFlow, originally provided at <a href="https://github.com/huzi96/Coarse2Fine-ImaComp">Coarse2Fine-ImaComp</a>. Besides, this version contains several improvements over the original one:

1. We have a brand new arithmetic coder implementation (in C++). It makes the encoding and decoding significantly faster (~10 times and more).
2. We now have full support of GPU accelerated encoding and decoding. It can be toggled by "--device cuda".
3. Partitioning is implemented, providing the support of compressing and decompressing images in GPUs with limited memory.

These new features are still being tested. If you encounter any problem, please feel free to contact me.

## Running
Before running the python script, you need to compile the arithmetic coder, with:

```g++ module_arithmeticcoding.cpp -o module_arithmeticcoding```

You may first download the trained weights from <a href="https://drive.google.com/open?id=1QL9lpEeTgzJMCEZ2m-9gOxGr6TChB2PU">Google Drive</a> and place the ```.pk``` files under the ```models``` folder (that is, to make ```'./models/model0_qp1.pk``` exist).

### Help
```python AppEncDec.py -h```
### Encoder (GPU Mode)
```python AppEncDec.py compress example.png example.bin --qp 1 --model_type 0 --device cuda```

### Decoder (GPU Mode)
```python AppEncDec.py decompress example.bin example_dec.png --device cuda```

Detailed command line options are documented in the ```help``` mode of the APP.
