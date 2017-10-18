import numpy as np
import tensorflow as tf
import os,time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

n=1000
start_time = time.time()
for i in range(10):
    w = np.random.rand(n,n)
    w = tf.constant(w)
    with tf.Session() as sess:
        tf.matmul(w,w).eval()
elapsed_time = time.time() - start_time
print(elapsed_time)
