#!/usr/bin/env python

#    Copyright 2021 Abdelfattah TOULAOUI
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Foobar is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Foobar.  If not, see <https://www.gnu.org/licenses/>.

import pandas
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
import os
from os import path
import numpy as np
import cv2

import random
import sys

# Take the triplet CSV
df = pandas.read_csv('dataset/training_triplet.csv')

# Training images and testing images
training = df.iloc[:4000000]
testing  = df.iloc[4000000:]

initializer = tf.keras.initializers.HeNormal()

# The model for embedding the faces into a vector
def face_vectorizer(vsize=4):
    cnn = tf.keras.applications.ResNet50V2(input_shape=(128, 128, 1), include_top=False,
                                           weights=None, pooling='max')
    x_in = layers.Input(shape=(128, 128, 1))
    x = cnn(x_in)
    x = layers.Flatten()(x)
    x = layers.Dense(vsize*256,
                     kernel_initializer=initializer,
                     bias_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(vsize*192,
                     kernel_initializer=initializer,
                     bias_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(vsize*128,
                     kernel_initializer=initializer,
                     bias_initializer=initializer)(x)
    return tf.keras.Model(inputs=x_in, outputs=x)

# The model to compare the facial embeddings
def face_comparator(vsize=4):
    x1_in = layers.Input(shape=(vsize*128,))
    x2_in = layers.Input(shape=(vsize*128,))

    x = layers.Subtract()([x1_in, x2_in])
    
    x = layers.Lambda(lambda x: tf.reduce_sum(tf.square(x), axis=1))(x)

    return keras.Model([x1_in, x2_in], x, name='face_comparator')

# A model to put the two previous models together
def face_recog(v, c):
    x1_in = layers.Input(shape=(128, 128, 1))
    x2_in = layers.Input(shape=(128, 128, 1))
    x3_in = layers.Input(shape=(128, 128, 1))
    
    x1 = v(x1_in)
    x2 = v(x2_in)
    x3 = v(x3_in)
    
    p = c([x1, x2])
    n = c([x1, x3])
    
    return tf.keras.Model([x1_in, x2_in, x3_in], [p, n], name='face_recog')


# Metrics to keep track of during training
def vmodel_acc(y_pred1, y_pred2, margin=1.):
    return (true_positives(y_pred1, margin)+true_negatives(y_pred2, margin))/2

def true_positives(y_pred, margin=1.):
    true_positives = tf.cast(tf.less(y_pred, margin), y_pred.dtype)
    return true_positives

def true_negatives(y_pred, margin=1.):
    true_negatives = tf.cast(tf.greater_equal(y_pred, margin), y_pred.dtype)
    return true_negatives

# Siamese network custom model
class SiameseModel(keras.Model):
    def __init__(self, siamese_network, margin=1.):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.Mean(name="acc")
        self.tpos_tracker = keras.metrics.Mean(name="true_positives")
        self.tneg_tracker = keras.metrics.Mean(name="true_negatives")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # Train and optimize the model parameters
        with tf.GradientTape() as tape:
            ap_distance, an_distance = self.siamese_network(data)
            loss = self._compute_loss(ap_distance, an_distance)
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Update and return the loss metric
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(vmodel_acc(ap_distance, an_distance, self.margin))
        self.tpos_tracker.update_state(true_positives(ap_distance, self.margin))
        self.tneg_tracker.update_state(true_negatives(an_distance, self.margin))
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Predict and calculate the loss
        ap_distance, an_distance = self.siamese_network(data)
        loss = self._compute_loss(ap_distance, an_distance)

        # Update and return the loss metric.
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(vmodel_acc(ap_distance, an_distance, self.margin))
        self.tpos_tracker.update_state(true_positives(ap_distance, self.margin))
        self.tneg_tracker.update_state(true_negatives(an_distance, self.margin))
        return {m.name: m.result() for m in self.metrics}

    def _compute_loss(self, ap_distance, an_distance):
        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        # For our purposes, the true negatives are more important than true positives
        loss = ap_distance + tf.maximum(self.margin - an_distance, 0.0) * 10
        return loss

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker,
                self.tpos_tracker, self.tneg_tracker]



# The pipeline that feeds the input images 
# to the model
IMAGE_PATH = 'dataset/images/'

# Create the model output directory if it doesn't exist
os.makedirs('nn', exist_ok=True)

# Create the model
vectorizer = face_vectorizer()
vectorizer.summary()
comparator = face_comparator()
comparator.summary()
model = face_recog(vectorizer, comparator)
model = SiameseModel(model, .5)
model.compile(
        optimizer=keras.optimizers.Adam(1e-5)
        )

# Write the model config to files
with open('nn/vectorizer.json', 'w') as f:
    f.write(vectorizer.to_json())

# Define the image preprocessing pipeline
def preprocess_image(image):
    img = tf.io.decode_image(tf.io.read_file(IMAGE_PATH + image), channels=3)
    return tf.image.rgb_to_grayscale(img)

def to_dataset(df, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(tuple(df[f].values
                                                   for f in ['file1', 'file2', 'file3']))
    dataset = dataset.map(lambda x1, x2, x3: tuple(preprocess_image(i) for i in [x1, x2, x3]))
    dataset = dataset.repeat().batch(batch_size)
    return dataset


# Define the training and test datasets
train_generator = to_dataset(training, 16)
test_generator = to_dataset(testing, 16)


# Reduce the learning rate on plateau, this will help our model keep learning 
# even if it hits a wall using any given learning rate
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.85,
                              verbose=2, patience=5, min_lr=0.)

# Finally train the model
model.fit(train_generator,
        steps_per_epoch=1250,
        validation_data=test_generator,
        validation_steps=40,
        epochs=100,
        callbacks=[reduce_lr])

# Save the finished model
vectorizer.save('nn/vectorizer.h5')

