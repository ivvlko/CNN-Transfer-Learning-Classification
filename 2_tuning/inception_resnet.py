import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf


from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input , Dense, Flatten, Dropout, Conv2D, MaxPool2D

from tensorflow.keras.applications import resnet_v2, inception_resnet_v2

import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.metrics import TopKCategoricalAccuracy



### HERE WE JUST PREPARE THE DS THE SAME WAY WE DID BEFORE

dataset, info = tfds.load(name="stanford_dogs", with_info=True)
resnet_inception = inception_resnet_v2.InceptionResNetV2()

IMG_LEN = 299
IMG_SHAPE = (IMG_LEN, IMG_LEN,3)
N_BREEDS = 120

training_data = dataset['train']
test_data = dataset['test']

def preprocess(ds_row):
  
    image = tf.image.convert_image_dtype(ds_row['image'], dtype=tf.float32)
    image = tf.image.resize(image, (IMG_LEN, IMG_LEN), method='nearest')
  
    label = tf.one_hot(ds_row['label'],N_BREEDS)

    return image, label

def prepare(dataset, batch_size=None):
    ds = dataset.map(preprocess, num_parallel_calls=4)
    ds = ds.shuffle(buffer_size=1000)
    if batch_size:
      ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


resnet_inception = Model(inputs = resnet_inception.layers[0].input, outputs = resnet_inception.layers[780].output)
for layer in resnet_inception.layers:
  layer.trainable = False

data_augmentation = tf.keras.Sequential(
  [
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(299, 
                                                              299,
                                                              3)),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)


def plot_train_and_validation_results(train, validation, title, epochs_range):
  plt.figure(figsize=[6,4])
  plt.plot(train)
  plt.plot(validation)
  plt.xlabel('Epochs')
  plt.ylabel('Value')
  if epochs_range:
    plt.xticks(epochs_range)
  plt.title(title)
  plt.legend(['Training', 'Validation'])
  plt.show()

# Architectures to test

resnet_inception_one_fnn_layer = Sequential([
                           data_augmentation,
                           resnet_inception,
                           Dense(N_BREEDS, activation = 'softmax')
])

resnet_inception_two_fnn_layers = Sequential([
                           data_augmentation,
                           resnet_inception,
                           Dense(256, activation = 'relu'),
                           Dropout(0.2),
                           Dense(N_BREEDS, activation = 'softmax')
])

resnet_inception_three_fnn_layers = Sequential([
                           data_augmentation,
                           resnet_inception,
                           Dense(512, activation = 'relu'),
                           Dropout(0.3),
                           Dense(256, activation = 'relu'),
                           Dropout(0.1),
                           Dense(N_BREEDS, activation = 'softmax')
])



### TUNING STARTS HERE

# We'll focus mainly on the most important hyperparameters since we are using Transfer Learning with very strong model 
OPTIMIZERS = [Adam, SGD, RMSprop]
LEARNING_RATE = [0.01, 0.001, 0.0001, 0.00001, 0.03, 0.003, 0.0003, 0.00003, 0.05, 0.005, 0.0005, 0.00005]
BATCH_SIZE = [16, 32, 64, 128]

early_stopping = EarlyStopping(monitor = 'val_loss', patience = 3)

# Grid Search

def perform_grid_search(opt, lr, bs, model):
    # Just cross tunning every parameter with each other with 3 nested for loops.
    for batch in bs:
        train_batches = prepare(training_data, batch_size=batch)
        test_batches = prepare(test_data, batch_size=batch)
        for optimizer in opt:
            for learning_rate in lr:
                model.compile(optimizer = optimizer(learning_rata = learning_rate), loss = 'categorical_crossentropy', metrics = ['accuracy', TopKCategoricalAccuracy(k=3)])
                model_history = model.fit(train_batches, validation_data=test_batches,  epochs = 50, callbacks = [early_stopping])
                plot_train_and_validation_results(model_history.history['loss'], model_history.history['val_loss'], 'Training and Validation Loss', range(0, 51, 10) )
                plot_train_and_validation_results(model_history.history['accuracy'], model_history.history['val_accuracy'], 'Training and Validation Accuracy', range(0, 51, 10) )
                plot_train_and_validation_results(model_history.history['top_k_categorical_accuracy'], model_history.history['val_top_k_categorical_accuracy'], 'Top 3 Training and Validation Accuracy',  range(0, 51, 10))


perform_grid_search(OPTIMIZERS, LEARNING_RATE, BATCH_SIZE, resnet_inception_one_fnn_layer)
perform_grid_search(OPTIMIZERS, LEARNING_RATE, BATCH_SIZE,resnet_inception_two_fnn_layers)
perform_grid_search(OPTIMIZERS, LEARNING_RATE, BATCH_SIZE,resnet_inception_three_fnn_layers)
