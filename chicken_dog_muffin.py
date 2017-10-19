import os
import random
import math
import csv
import tensorflow as tf
import numpy as np
import glob
# we use The Python Imaging Library (PIL) to preprocess image data
# you can install it by: pip install Pillow
from PIL import Image

img_size = 64
batch_size = 50
# number of features on each layer
fea_conv1 = 32
fea_conv2 = 64
fea_fc1 = 1024

# read images and transform them into numpy array
train_path = glob.glob("./data/training_data/images/*.jpg")
test_path = glob.glob("./data/test_data/images/*.jpg")

train_img = []
test_img = []

for file_name in train_path:
    pil_im = Image.open(file_name).convert('RGB')
    pil_resize = pil_im.resize((img_size,img_size))
    pil_array = np.array(pil_resize.getdata()).reshape(img_size, img_size, 3)
    train_img.append(pil_array)

for file_name in test_path:
    pil_im = Image.open(file_name).convert('RGB')
    pil_resize = pil_im.resize((img_size,img_size))
    pil_array = np.array(pil_resize.getdata()).reshape(img_size, img_size, 3)
    test_img.append(pil_array)

train_img = np.array(train_img)
test_img = np.array(test_img)
print('Images load is done!')

# read labels and transform them into 3000x3 numpy array
train_label = np.zeros((3000, 3))
test_label = np.zeros((3000, 3))
with open('./data/training_data/label_train.csv') as csvfile:
    label_train = csv.reader(csvfile, delimiter=',')
    ## delete row names
    next(label_train)
    train_list = list(label_train)

with open('./data/test_data/label_test.csv') as csvfile:
    label_test = csv.reader(csvfile, delimiter=',')
    next(label_test)
    test_list = list(label_test)
    
for i in range(3000):
    j_1 = int(train_list[i][1])
    j_2 = int(test_list[i][1])
    train_label[i, j_1] = 1
    test_label[i, j_2] = 1
print('Label loading and transform is done!')

# weight initialization

## values whose magnitude larger than 2 standard 
## deviation would be dropped and repicked
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

## initialize them with a slightly positive initial bias 
## to avoid "dead neurons"
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# convolution and pooling
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1],
                       padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], 
                          strides=[1, 2, 2, 1], padding='SAME')


# first convolutional layer
W_conv1 = weight_variable([5, 5, 3, fea_conv1])
b_conv1 = bias_variable([fea_conv1])


x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 3])

x_image = x

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolutional layer
W_conv2 = weight_variable([5, 5, fea_conv1, fea_conv2])
b_conv2 = bias_variable([fea_conv2])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# densily connected layer
size_conv2 = int(img_size/4)
W_fc1 = weight_variable([size_conv2 * size_conv2 * fea_conv2, fea_fc1])
b_fc1 = bias_variable([fea_fc1])

h_pool2_flat = tf.reshape(h_pool2, [-1, size_conv2 * size_conv2 * fea_conv2])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer
W_fc2 = weight_variable([fea_fc1, 3])
b_fc2 = bias_variable([3])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# train and evaluate the model
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# train and evaluate the model
id_raw = np.arange(3000)
it_per_epo = math.floor(3000/batch_size)
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        num_in_epo = i % it_per_epo
        if num_in_epo == 0:
            # if already run an epoch
            # reshuffle the training data
            id_shuf = random.sample(list(id_raw), 3000)
            print('Data reshuffled!')
            batch_img = train_img[id_shuf[:batch_size]]
            batch_label = train_label[id_shuf[:batch_size]]
            train_step.run(feed_dict={x: batch_img, y_: batch_label, keep_prob: 0.5})
            train_accuracy = accuracy.eval(feed_dict={
                x: batch_img, y_: batch_label, keep_prob: 1.0})
        else:
            batch_img = train_img[id_shuf[(num_in_epo*batch_size):(num_in_epo*batch_size + batch_size)]]
            batch_label = train_label[id_shuf[(num_in_epo*batch_size):(num_in_epo*batch_size + batch_size)]]
            train_step.run(feed_dict={x: batch_img, y_: batch_label, keep_prob: 0.5})
            train_accuracy = accuracy.eval(feed_dict={
                x: batch_img, y_: batch_label, keep_prob: 1.0})
        if i%10 ==0:
            print('step %d, training accuracy %g' % (i, train_accuracy))
            print('test accuracy %g' % accuracy.eval(feed_dict={
                x: test_img, 
                y_: test_label, keep_prob: 1.0}))



