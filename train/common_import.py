# GPU無効化
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import applications
from tensorflow.keras import optimizers
from tensorflow.python.keras import callbacks
from tensorflow.python.keras import metrics
from tensorflow.python.keras.datasets import cifar10
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
import signal
from pathlib import Path
import glob
import cv2
import pickle
import re
import time
import os
import random
from random import shuffle
from itertools import islice
from pprint import pprint

from defined_models import efficientnetv2
from ImageIterator import *
from ImageSequenceIterator import *



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


###サンプルデータ取得###
def getSampleData():
    # CIFAR10データの読み込み
    (X_train, y_train), (X_test, y_test), = cifar10.load_data()
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    # 0番目のクラスと1番目のクラスのデータを結合
    X_train = np.concatenate([X_train[y_train == 0], X_train[y_train == 1]], axis=0)
    y_train = np.concatenate([y_train[y_train == 0], y_train[y_train == 1]], axis=0)
    X_test = np.concatenate([X_test[y_test == 0], X_test[y_test == 1]], axis=0)
    y_test = np.concatenate([y_test[y_test == 0], y_test[y_test == 1]], axis=0)

    # Generator生成
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(
        rescale=1. / 255
    )
    train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
    test_generator = test_datagen.flow(X_test, y_test, batch_size=32)

    return (train_generator,test_generator)


###評価関数取得###
def getMetrics(mode=None):
    mode = "all"
    metrics_list = []
    metrics_list.append('accuracy')
    # metrics_list.append(metrics.Accuracy())
    if mode!="accuracy":
        metrics_list.append(metrics.AUC())
    if mode=="all":
        metrics_list.append(metrics.Precision())
        metrics_list.append(metrics.Recall())
    if mode!="accuracy":
        metrics_list.append(metrics.TruePositives())
        metrics_list.append(metrics.TrueNegatives())
        metrics_list.append(metrics.FalsePositives())
        metrics_list.append(metrics.FalseNegatives())
    return metrics_list








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
        # model.add(layers.Conv2D(512, (3, 3), activation='relu'))
        # model.add(layers.Conv2D(512, (3, 3), activation='relu'))
        # model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
            metrics=getMetrics()
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
        model.add(layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        model.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(units=4096,activation="relu"))
        model.add(layers.Dense(units=4096,activation="relu"))
        model.add(layers.Dense(units=1, activation="sigmoid"))

        # model = models.Sequential(name="VGG16")
        # model.add(applications.vgg16.VGG16(include_top=False, weights=None, input_shape=input_shape))
        # model.add(layers.Flatten())
        # model.add(layers.Dense(256, activation='relu'))
        # model.add(layers.Dropout(0.5))
        # model.add(layers.Dense(1, activation='softmax'))

        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
            metrics=getMetrics("all")
        )
    return model


### CNN Xception ###
def loadXception(input_shape=(480,640,3),gpu_count=2):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = models.Sequential(name="Xception")
        model.add(applications.xception.Xception(include_top=False, weights=None, input_shape=input_shape))
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
            metrics=getMetrics("all")
        )
    return model

### CNN EfficientNetV2 ###
def loadEfficientNetV2(input_shape=(480,640,3),gpu_count=2):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = models.Sequential(name="EfficientNetV2")
        model.add(layers.InputLayer(input_shape=input_shape))
        model.add(efficientnetv2.effnetv2_model.get_model('efficientnetv2-b0', include_top=False))
        model.add(layers.Dropout(rate=0.2))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
            metrics=getMetrics("all")
        )
    return model


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
def loadSampleRnn(input_shape=(5,256,256,3),gpu_count=2):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        inputs = layers.Input(shape=input_shape)
        x0 = layers.ConvLSTM2D(filters=16, kernel_size=(3,3), padding="same", return_sequences=True, data_format="channels_last")(inputs)
        x0 = layers.BatchNormalization(momentum=0.6)(x0)
        x0 = layers.ConvLSTM2D(filters=16, kernel_size=(3,3), padding="same", return_sequences=True, data_format="channels_last")(x0)
        x0 = layers.BatchNormalization(momentum=0.8)(x0)
        x0 = layers.ConvLSTM2D(filters=3, kernel_size=(3,3), padding="same", return_sequences=False, data_format="channels_last")(x0)
        x0 = layers.Flatten()(x0)
        # x0 = layers.Dense(4096, activation='relu')(x0)
        # x0 = layers.Dense(2048, activation='relu')(x0)
        x0 = layers.Dense(512, activation='relu')(x0)
        x0 = layers.Dense(128, activation='relu')(x0)
        output = layers.Dense(1, activation='relu')(x0)
        # output = layers.Activation('tanh')(x0)
        model = models.Model(inputs=inputs, outputs=output)
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
            metrics=[metrics.Accuracy(), metrics.AUC(), metrics.Precision(), metrics.Recall() , metrics.TruePositives(), metrics.TrueNegatives(), metrics.FalsePositives(), metrics.FalseNegatives()]
        )
    return model


