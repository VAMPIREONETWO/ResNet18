import tensorflow as tf
from tensorflow import keras
from ResNet18 import ResNet18
from keras.datasets import cifar10

using_gpu_index = 0
gpu_list = tf.config.experimental.list_physical_devices('GPU')
if len(gpu_list) > 0:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpu_list[using_gpu_index],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)]  # limit the size of GPU memory
        )
    except RuntimeError as e:
        print(e)
else:
    print("Got no GPUs")

# data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255
x_test = x_test / 255

# model construction
model = ResNet18(10, in_shape=(32, 32, 3), pre_filter_size=5)
model.build(input_shape=(None, 32, 32, 3))
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

# train and predict
model.fit(x_train, y_train, epochs=30, batch_size=16)
y_eva = model.evaluate(x_test, y_test, return_dict=True)

# accuracy: 0.7743
