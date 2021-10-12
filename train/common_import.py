from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import applications
from tensorflow.keras import optimizers
from tensorflow.python.keras import callbacks
from tensorflow.python.keras import metrics
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
import tensorflow.python.keras.backend as K
# from sklearn.metrics import roc_curve
# from sklearn.metrics import auc
# from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
#import cv2
import numpy as np
from pathlib import Path
import cv2
import pickle
import re
import time
import os
from itertools import islice



### GPU稼働確認 ###
import tensorflow as tf
from tensorflow.python.client import device_lib
print(tf.__version__)
print(device_lib.list_local_devices())


### ROC AUC ###
# def auc(y_true, y_pred):
#     auc = tf.metrics.auc(y_true, y_pred)[1]
#     K.get_session().run(tf.local_variables_initializer())
#     return auc

# def roc_auc(y_true, y_pred):
#     roc_auc = tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)
#     return roc_auc

# def roc(y_true, y_pred):
#     fpr, tpr, thresholds = roc_curve(y_true, y_pred)
#     return fpr, tpr, thresholds




### CNN 2値分類 ###
def loadSampleCnn(input_shape=(480,640,3),gpu_count=2):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = models.Sequential(name="SampleCNN")
        model.add(layers.Conv2D(16, (3, 3), activation='relu', data_format='channels_last', input_shape=input_shape))
        model.add(layers.Conv2D(16, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(256, (3, 3), activation='relu'))
        model.add(layers.Conv2D(256, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(512, (3, 3), activation='relu'))
        model.add(layers.Conv2D(512, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
            metrics=[metrics.Accuracy(), metrics.AUC(), metrics.Precision(), metrics.Recall() , metrics.TruePositives(), metrics.TrueNegatives(), metrics.FalsePositives(), metrics.FalseNegatives()]
        )
    return model


### CNN VGG16 ###
def loadVgg16(input_shape=(480,640,3),gpu_count=2):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        # input_tensor = models.Input(shape=input_shape)
        # vgg16 = applications.vgg16.VGG16(include_top=False, weights=None, input_tensor=input_tensor)
        # _model = models.Sequential()
        # _model.add(layers.Flatten(input_shape=vgg16.output_shape[1:]))
        # _model.add(layers.Dense(256, activation='relu'))
        # _model.add(layers.Dropout(0.5))
        # _model.add(layers.Dense(1, activation='softmax'))
        # model = models.Model(inputs=vgg16.input, outputs=_model(vgg16.output), name="VGG16")
        # # for layer in model.layers[:15]:
        # #     layer.trainable = False

        model = models.Sequential(name="VGG16")
        model.add(layers.Conv2D(input_shape=input_shape,filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(units=4096,activation="relu"))
        model.add(layers.Dense(units=4096,activation="relu"))
        model.add(layers.Dense(units=1, activation="softmax"))

        # model = models.Sequential(name="VGG16")
        # model.add(applications.vgg16.VGG16(include_top=False, weights=None, input_shape=input_shape))
        # model.add(layers.Dense(256, activation='relu'))
        # model.add(layers.Dropout(0.5))
        # model.add(layers.Dense(1, activation='softmax'))

        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
            metrics=[metrics.Accuracy(), metrics.AUC(), metrics.Precision(), metrics.Recall() , metrics.TruePositives(), metrics.TrueNegatives(), metrics.FalsePositives(), metrics.FalseNegatives()]
        )
    return model


### CNN Xception ###
def loadXception(input_shape=(480,640,3),gpu_count=2):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = models.Sequential(name="Xception")
        model.add(applications.xception.Xception(include_top=False, weights=None, input_shape=input_shape))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, activation='softmax'))
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
            metrics=[metrics.Accuracy(), metrics.AUC(), metrics.Precision(), metrics.Recall() , metrics.TruePositives(), metrics.TrueNegatives(), metrics.FalsePositives(), metrics.FalseNegatives()]
        )
    return model

### CNN EfficientNetV2 ###
def loadEfficientNetV2(input_shape=(480,640,3),gpu_count=2):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        pass
    return None


### CNN AutoEncoder ###
def loadAutoEncoder(input_shape=(480,640,3),gpu_count=2):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        input_img = models.Input(shape=input_shape)
        x = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(input_img)
        x = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        # x = layers.ZeroPadding2D(padding=((2, 2), (0, 0)), data_format=None)(x)
        x = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        y = layers.Conv2D(3, (3, 3), activation='sigmoid', kernel_initializer='he_uniform', padding='same')(x)
        model = models.Model(inputs=input_img, outputs=y)
        model.compile(
            loss='mean_squared_error',
            optimizer=optimizers.Adam(lr=1e-4),
            metrics=[metrics.Accuracy()]
        )
    return model



### RNN ###
def loadRnn(input_shape=(5,256,256,3),gpu_count=2):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        inputs = layers.Input(shape=input_shape)
        x0 = layers.ConvLSTM2D(filters=16, kernel_size=(3,3), padding="same", return_sequences=True, data_format="channels_last")(inputs)
        x0 = layers.BatchNormalization(momentum=0.6)(x0)
        x0 = layers.ConvLSTM2D(filters=16, kernel_size=(3,3), padding="same", return_sequences=True, data_format="channels_last")(x0)
        x0 = layers.BatchNormalization(momentum=0.8)(x0)
        x0 = layers.ConvLSTM2D(filters=3, kernel_size=(3,3), padding="same", return_sequences=False, data_format="channels_last")(x0)
        output = layers.Activation('tanh')(x0)
        model = models.Model(inputs=inputs, outputs=output)
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
            metrics=[metrics.Accuracy(), metrics.AUC(), metrics.Precision(), metrics.Recall() , metrics.TruePositives(), metrics.TrueNegatives(), metrics.FalsePositives(), metrics.FalseNegatives()]
        )
    return model


