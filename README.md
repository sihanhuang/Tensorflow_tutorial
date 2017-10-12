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







