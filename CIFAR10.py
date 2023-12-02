
from tensorflow import keras
from keras import layers, Sequential, Model
from ResNet18 import ResNet18
from keras.datasets import cifar10
import numpy as np
import pandas as pd


# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(x_train.shape)
# print(y_train.shape)
# print(x_train)
# x_train = np.array(x_train)
# print(type(x_train))
# s = np.shape(x_train)
# print(s)

# model = Sequential()
# model.add(layers.Conv2D(64,(7,7),2,padding="same"))
# model.build(input_shape=(None,32,32,3))
# model.compile(optimizer="adam",loss="categorical_crossentropy")
# model.summary()
# model.fit(x_train,y_train,epochs=10,batch_size=100)
# y_pre = model.predict(x_test)
# print(y_pre)
# print(y_pre.shape())
model = ResNet18(10)
model.build(input_shape=(None,32,32,3))
model.compile(optimizer="adam",loss="categorical_crossentropy")
model.summary()