## Train a Coarse-to-Fine Compression Model

### Data
We use DIV2K as the training data. All 800 images are included. Besides, we down-sampled all images to half of their sizes and build a training set with 1600 images.

You may place the 1600 PNG files in a directory, assumed to be named ```IMAGE_PATH```.

### Command
An example command to train the network with 2 GPUs is as follows,
```CUDA_VISIBLE_DEVICES=0,1 python train.py train --batchsize 16 --train_glob "IMAGE_PATH/*.png" --checkpoint_dir checkpoint --lambda 0.004```

You may use different numbers of GPUs. Please modify ```n_parallel``` in the code if you use a different number of GPUs.

The training checkpoints will be saved in ```checkpoint``` as set.

Note that this is an experimental implementation of the training procedure with PyTorch. The code would train the model from scratch (without pretraining) and the resulting models would even achieve better R-D performance than released in the paper. If you encounter any problem, please feel free to contact me.
