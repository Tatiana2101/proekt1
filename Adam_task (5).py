#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# In[7]:


from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Conv2D, Flatten, Dense, AvgPool2D
from tensorflow.keras.models import Sequential
import numpy as np
from tensorflow.keras.optimizers import Adam


# In[16]:


def load_train(path):
    features_train = np.load(path + 'train_features.npy')
    target_train = np.load(path + 'train_target.npy')
    features_train = features_train.reshape(-1, 28, 28, 1) / 255.0   
    return features_train, target_train 


# In[14]:


def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(filters=6, kernel_size=(5, 5), padding='same',
                     activation="relu", input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    optimizerAd = Adam(learning_rate=0.1)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizerAd,
                      metrics=['acc'])
    return model 


# In[15]:


def train_model(model, train_data, test_data, batch_size=16, epochs=5,
                steps_per_epoch=None, validation_steps=None):

    features_train, target_train = train_data
    features_test, target_test = test_data
    model.fit(features_train, target_train, 
              validation_data=(features_test, target_test),
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2, shuffle=True)

    return model 

