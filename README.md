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
Load the cuda module.
```bash
$ module load cuda80/toolkit cuda80/blas cudnn
```
Load anaconda:
```bash
$ module load anaconda
```
Install tensorflow and tensorflow-gpu (this can take few minutes, please wait - you need to do it only once):
```bash
$ pip install tensorflow
$ pip install tensorflow-gpu 
```

### Use TensorFlow on Habanero (3 ways)
#### 1. Start Python directly
```bash
$ python
Python 3.5.2 |Anaconda 4.2.0 (64-bit)| (default, Jul  2 2016, 17:53:06) 
```

```python
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
```
You set it up successfully if you see the following output.
```
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

#### 2. Run an existed Python script
To do this, simply use
```bash
$ python Hello.py
```
Does your CUDA application need to target a specific GPU? If you are writing GPU enabled code, you would typically use a device query to select the desired GPUs. However, a quick and easy solution for testing is to use the environment variable ```CUDA_VISIBLE_DEVICES``` to restrict the devices that your CUDA application sees. This can be useful if you are attempting to share resources on a node or you want your GPU enabled executable to target a specific GPU.
```bash
$ CUDA_VISIBLE_DEVICES=1 python Hello.py
```

#### 3. Submitting Jobs
If you want to submit the job using a bash file, you can use
```bash
sbatch gpu.sh
```

### Monitoring GPU devices
The NVIDIA System Management Interface (nvidia-smi) is a command line utility, based on top of the NVIDIA Management Library (NVML), intended to aid in the management and monitoring of NVIDIA GPU devices. Firstly, create a new screen under the same session:
```bash
$ screen
```
Then you can use ```Ctrl+a+c``` to create another screen. You can also use ```Ctrl+a+n``` or ```Ctrl+a+p``` to switch among screens. To monitor your GPU devices, you should run
```bash
$ nvidia-smi
```
in a new screen when you run your program.
