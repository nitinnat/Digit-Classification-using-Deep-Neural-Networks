
# coding: utf-8

# In[2]:

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
import pandas as pd
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import matplotlib.image as mpimg
import tensorflow as tf
from tqdm import tqdm
# Config the matplotlib backend as plotting inline in IPython
get_ipython().magic(u'matplotlib inline')

data_root = '.' # Change me to store data elsewhere
all_data = pickle.load(open(os.path.join(data_root, 'notMNIST.pickle')))
train_dataset = all_data["train_dataset"]
train_labels = all_data["train_labels"]
valid_dataset = all_data["valid_dataset"]
valid_labels = all_data["valid_labels"]
test_dataset = all_data["test_dataset"]
test_labels = all_data["test_labels"]
num_labels = 10
train_labels = (np.arange(num_labels) == train_labels[:,None]).astype(np.float32)
valid_labels = (np.arange(num_labels) == valid_labels[:,None]).astype(np.float32)
test_labels = (np.arange(num_labels) == test_labels[:,None]).astype(np.float32)
del all_data #to save memory
print(train_dataset.shape, valid_dataset.shape, test_dataset.shape)
print(train_labels.shape, valid_labels.shape,test_labels.shape)
image_size = 28
num_channels = 1


# In[3]:

train_dataset = np.reshape(train_dataset, [-1,image_size,image_size,num_channels]).astype(np.float32)
valid_dataset = np.reshape(valid_dataset, [-1,image_size,image_size, num_channels]).astype(np.float32)
test_dataset = np.reshape(test_dataset, [-1, image_size,image_size,num_channels]).astype(np.float32)


# In[4]:

def calc_output_size(input_size, filter_size, stride, padding):
    if padding == "same":
        pad = -1.00
    else:
        pad = 0.00
    output = (input_size - filter_size - 2*pad) / stride
    return output
    

batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64

graph = tf.Graph()

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  layer1_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, num_channels, depth], stddev=0.1))
  layer1_biases = tf.Variable(tf.zeros([depth]))
  layer2_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth, depth], stddev=0.1))
  layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
  final_image_size  = calc_output_size(image_size, patch_size,2,padding = 'same')
  final_image_size = calc_output_size(final_image_size,patch_size,2, padding = 'same')
  layer3_weights = tf.Variable(tf.truncated_normal(
      [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
  layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
  layer4_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, num_labels], stddev=0.1))
  layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
  
  # Model.
  def model(data):
    conv1 = tf.nn.conv2d(data, layer1_weights,strides = [1,2,2,1],padding = 'SAME')
    hidden1 = tf.nn.relu(conv1 + layer1_biases)
    pool1 = tf.nn.max_pool(hidden1, strides = [1,2,2,1],ksize = [1,2,2,1], padding = 'SAME')
    
    conv2 = tf.nn.conv2d(pool1, layer2_weights, strides = [1,2,2,1], padding = 'SAME')
    hidden2 = tf.nn.relu(conv2 + layer2_biases)
    pool2 = tf.nn.max_pool(hidden2, strides = [1,2,2,1], ksize = [1,2,2,1], padding = 'SAME')
    
    shape = hidden2.get_shape().as_list()
    reshape = tf.reshape(hidden2, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    return tf.matmul(hidden, layer4_weights) + layer4_biases
  
  # Training computation.
  logits = model(tf_train_dataset)
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  test_prediction = tf.nn.softmax(model(tf_test_dataset))


# In[26]:



def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])



num_steps = 10001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

