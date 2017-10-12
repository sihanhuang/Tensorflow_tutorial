# Tensorflow_tutorial

### Pip installation

Start a terminal (a shell). You'll perform all subsequent steps in this shell.
```bash
# Mac OS X
$ sudo easy_install pip
$ sudo easy_install --upgrade six
```
You can simply install tensorflow on Linux, Mac or Windows with pip install. Note you will need pip version 8.1 or later for the following commands to work on Linux :
```bash
$ pip install tensorflow
$ pip install tensorflow-gpu
```

### Import tensorflow on GPU node

First request a GPU node on Habanero (use "stats" if you are a member of "stats" group, otherwise use your group name):
```bash
$ srun --pty -t 0-02:00:00 --gres=gpu:1 -A stats /bin/bash
```
Load these modules:
```bash
$ module load cuda80/toolkit cuda80/blas cudnn/6.0_8
$ module load anaconda/2-4.2.0
```
Install tensorflow-gpu as user (this can take few minutes, please wait - you need to do it only once):
```bash
$ pip install tensorflow-gpu --user
```
Start python and test tensorflow
```bash
$ python
Python 2.7.12 |Anaconda 4.2.0 (64-bit)| (default, Jul  2 2016, 17:42:40) 
```

```python
>>>import tensorflow as tf
>>>hello = tf.constant('Hello, TensorFlow!')
>>>sess = tf.Session()

name: Tesla K80
major: 3 minor: 7 memoryClockRate (GHz) 0.8235
pciBusID 0000:8a:00.0
Total memory: 11.20GiB
Free memory: 11.13GiB
2017-10-12 10:51:13.724565: I tensorflow/core/common_runtime/gpu/gpu_device.cc:976] DMA: 0 
2017-10-12 10:51:13.724600: I tensorflow/core/common_runtime/gpu/gpu_device.cc:986] 0:   Y 
2017-10-12 10:51:13.724645: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla K80, pci bus id: 0000:8a:00.0)

```





