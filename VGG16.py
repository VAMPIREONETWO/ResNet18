import tensorflow as tf
from keras.datasets import cifar10
from tensorflow import keras
from keras import layers, Sequential, Model
from keras.applications.vgg16 import VGG16


model16 = Sequential()
model16.add(
tf.keras.applications.vgg16.VGG16(
    weights='imagenet',
    include_top=False,
    classes=10,
    input_shape=(32,32,3)))
model16.add(layers.Flatten())
model16.add(layers.Dense(512, activation='relu', name='hidden1'))
model16.add(layers.Dropout(0.4))
model16.add(layers.Dense(256, activation='relu', name='hidden2'))
model16.add(layers.Dropout(0.4))
model16.add(layers.Dense(10, activation='softmax', name='predictions'))
model16.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
model16.summary()

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255
x_test = x_test / 255


# train
model16.fit(x_train, y_train, epochs=20, batch_size=16)
model16.evaluate(x_test, y_test, return_dict=True)