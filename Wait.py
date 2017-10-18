import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

for i in range(100):
    node = tf.constant(i, dtype=tf.float32)
    sess = tf.Session()
    print(sess.run([node]))
    time.sleep(5);
