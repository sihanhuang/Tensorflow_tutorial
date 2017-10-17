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
To validate your installation, enter the following short program inside the python interactive shell:
```python
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> print(sess.run(hello))
```

### Import tensorflow on Habanero

Load anaconda:
```bash
$ module load anaconda
```
Install tensorflow and tensorflow-gpu (this can take few minutes, please wait - you need to do it only once):
```bash
$ pip install tensorflow
$ pip install tensorflow-gpu 
```
Start python and test tensorflow
```bash
$ python
Python 2.7.12 |Anaconda 4.2.0 (64-bit)| (default, Jul  2 2016, 17:42:40) 
```

```python
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
```

If you don't want warnings, you can add this at the beginning
```python
>>> import os
>>> os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
```

``` python
>>> print(sess.run(hello))
```
You will see
```
Hello, TensorFlow!
```
