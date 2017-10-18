# Tensorflow_tutorial

### Install TensorFlow on your laptop

Start a terminal (a shell). You'll perform all subsequent steps in this shell.
```bash
# Mac OS X
$ sudo easy_install pip
```
You can simply install tensorflow on Linux, Mac or Windows with pip install. Note you will need pip version 8.1 or later for the following commands to work on Linux :
```bash
$ pip install tensorflow
$ pip install tensorflow-gpu
```
To validate your installation, enter the following short program inside the python interactive shell:
```python
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> print(sess.run(hello))
```

### Import TensorFlow on Habanero
To login interactively to a GPU node, run the following command, replacing stats with your account.
```bash
$ srun --pty -t 0-01:00 --gres=gpu:1 -A stats /bin/bash
```

Load anaconda:
```bash
$ module load anaconda
```
Load the cuda module.
```bash
$ module load cuda80/toolkit cuda80/blas cudnn
```
Load the cuda environment module which will add cuda to your PATH and set related environment variables. Note cuda 8.0 does not support gcc 6, so gcc 5 or earlier must be accessible in your environment when running nvcc.  
```bash 
$ module load gcc/4.8.5
```
Install tensorflow and tensorflow-gpu (this can take few minutes, please wait - you need to do it only once):
```bash
$ pip install tensorflow
$ pip install tensorflow-gpu 
```
Start python and test tensorflow
```bash
$ python
Python 3.5.2 |Anaconda 4.2.0 (64-bit)| (default, Jul  2 2016, 17:53:06) 
```

```python
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
Found device 0 with properties: 
name: Tesla K80
major: 3 minor: 7 memoryClockRate (GHz) 0.8235
pciBusID 0000:8a:00.0
Total memory: 11.20GiB
Free memory: 11.13GiB
2017-10-18 12:24:17.232879: I tensorflow/core/common_runtime/gpu/gpu_device.cc:976] DMA: 0 
2017-10-18 12:24:17.232917: I tensorflow/core/common_runtime/gpu/gpu_device.cc:986] 0:   Y 
2017-10-18 12:24:17.232984: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla K80, pci bus id: 0000:8a:00.0)
```

If there are warnings, you can add this at the beginning of your .py file:
```python
>>> import os
>>> os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
```

``` python
>>> print(sess.run(hello))
Hello, TensorFlow!
```
To run python script



### Submitting jobs

``` 
