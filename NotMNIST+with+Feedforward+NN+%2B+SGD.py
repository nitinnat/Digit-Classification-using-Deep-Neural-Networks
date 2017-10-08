
# coding: utf-8

# In[34]:

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


# In[35]:

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


# In[36]:

batch_size = 128
num_nodes = 1024
image_size = 28
beta = 0.01 #Regularization parameter
graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape = (batch_size, image_size* image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape =  (batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    weights_1 = tf.Variable(tf.truncated_normal([image_size*image_size, num_nodes]))
    biases_1 = tf.Variable(tf.zeros([num_nodes]))
    weights_2 = tf.Variable(tf.truncated_normal([num_nodes, num_labels]))
    biases_2 = tf.Variable(tf.zeros([num_labels]))
    #Let's create a random dropout array with probabilities the same length as that of the weights array
    #dropout_1 = np.random.uniform(size = weights_1.shape[0])
    #dropout_2 = np.random.uniform(size = weights_2.shape[0])
    
    
    logits_1 = tf.matmul(tf_train_dataset,weights_1) + biases_1
    relu_layer = tf.nn.relu(logits_1)
    # Dropout on hidden layer: RELU layer
    keep_prob = tf.placeholder("float")
    relu_layer_dropout = tf.nn.dropout(relu_layer, keep_prob)
    logits_2 = tf.matmul(relu_layer_dropout,weights_2) + biases_2

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits_2))
    #Find regularization to be added to the loss
    regularizers = tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2)
    loss = tf.reduce_mean(loss + beta * regularizers)
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    # Predictions for the training, validation, and test data.
    # These are not part of training, but merely here so that we can report
    # accuracy figures as we train.
    train_prediction = tf.nn.softmax(logits_2)
    
    #Valid prediction
    logits_1 = tf.matmul(tf_valid_dataset,weights_1) + biases_1
    relu_layer = tf.nn.relu(logits_1)
    logits_2 = tf.matmul(relu_layer,weights_2) + biases_2
    valid_prediction = tf.nn.softmax(logits_2)
    
    #Test prediction
    logits_1 = tf.matmul(tf_test_dataset,weights_1) + biases_1
    relu_layer = tf.nn.relu(logits_1)
    logits_2 = tf.matmul(relu_layer,weights_2) + biases_2
    test_prediction = tf.nn.softmax(logits_2)


# In[37]:

num_steps = 3001
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels,keep_prob :0.5}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

