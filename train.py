import numpy as np

import tensorflow as tf
from tensorflow import set_random_seed
import dataset
from matplotlib import pyplot as plt
from datetime import datetime
import cv2
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

mode = 'rbg' # tryb rbg/gray

set_random_seed(2)
batch_size = 20

# Prepare input data
classes = ['a', 'b', 'c', '0'] #, 'd', 'e', 'i', 'l', 'm', 'o', 'w', 'y'
num_classes = len(classes)

validation_size = 0.2
img_size = 256
num_channels = 3
train_path = 'training_data/rbg'
data = dataset.DataSet()
session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
validationToPlot = dict()
errorToPlot = dict()
lossToPlot = dict()
plotDate = "Plot date " + str(datetime.utcnow()).replace(".", "").replace(":", " ")


if(mode == 'gray'):
    num_channels = 1

## labels
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

##Network graph params
filter_size_conv1 = 3
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64

filter_size_conv4 = 3
num_filters_conv4 = 128

fc_layer_size = 256


def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def create_convolutional_layer(input,
                               num_input_channels,
                               conv_filter_size,
                               num_filters):

    ## We shall define the weights that will be trained using create_weights function.
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])

    ## We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)

    ## Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    ## We shall be using max-pooling.
    layer = tf.nn.max_pool(value=layer,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')
    layer += biases

    ## Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.relu(layer)

    return layer


def create_flatten_layer(layer):
    # We know that the shape of the layer will be [batch_size img_size img_size num_channels]
    # But let's get it from the previous layer.
    layer_shape = layer.get_shape()

    ## Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
    num_features = layer_shape[1:4].num_elements()

    ## Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(input,
                    num_inputs,
                    num_outputs,
                    use_relu=True):
    # Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)

    validationToPlot[epoch + 1] = acc
    errorToPlot[epoch + 1] = val_acc
    lossToPlot[epoch + 1] = val_loss

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(list(validationToPlot.keys()), list(validationToPlot.values()), color='g', label='Training Accuracy')
    ax.plot(list(errorToPlot.keys()), list(errorToPlot.values()), color='r', label='Validation Accuracy')
    ax.plot(list(lossToPlot.keys()), list(lossToPlot.values()), color='b', label='Validation Loss')
    ax.legend()
    ax.grid(True)
    fig.savefig('plots/' + plotDate + '.png')

    msg = "Training Batch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))


def train(num_iteration):
    global total_iterations

    for i in range(total_iterations,
                   total_iterations + num_iteration):

        x_batch, \
        y_true_batch, \
        cls_batch, \
        x_valid_batch, \
        y_valid_batch, \
        valid_cls_batch = \
            data.next_batch(batch_size, train_path, img_size, classes, validation_size)

        feed_dict_tr = {x: x_batch, y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch, y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_tr)

        if i % int(x_batch.shape[0] / batch_size) == 0:
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(x_batch.shape[0] / batch_size))

            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
            saver.save(session, ".\\network-model")

    total_iterations += num_iteration



layer_conv1 = create_convolutional_layer(input=x,
                                         num_input_channels=num_channels,
                                         conv_filter_size=filter_size_conv1,
                                         num_filters=num_filters_conv1)

layer_conv2 = create_convolutional_layer(input=layer_conv1,
                                         num_input_channels=num_filters_conv1,
                                         conv_filter_size=filter_size_conv2,
                                         num_filters=num_filters_conv2)

layer_conv3 = create_convolutional_layer(input=layer_conv2,
                                         num_input_channels=num_filters_conv2,
                                         conv_filter_size=filter_size_conv3,
                                         num_filters=num_filters_conv3)

layer_conv4 = create_convolutional_layer(input=layer_conv3,
                                         num_input_channels=num_filters_conv3,
                                         conv_filter_size=filter_size_conv4,
                                         num_filters=num_filters_conv4)

layer_flat = create_flatten_layer(layer_conv3)

layer_fc1 = create_fc_layer(input=layer_flat,
                            num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                            num_outputs=fc_layer_size,
                            use_relu=True)

layer_fc_mid = create_fc_layer(input=layer_fc1,
                            num_inputs=fc_layer_size,
                            num_outputs=fc_layer_size,
                            use_relu=True)

layer_fc2 = create_fc_layer(input=layer_fc_mid,
                            num_inputs=fc_layer_size,
                            num_outputs=num_classes,
                            use_relu=False)

saver = tf.train.Saver()

y_pred = tf.nn.softmax(layer_fc2, name='y_pred')
y_pred_cls = tf.argmax(y_pred, dimension=1)
session.run(tf.global_variables_initializer())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session.run(tf.global_variables_initializer())
total_iterations = 0
train(num_iteration=5000)
