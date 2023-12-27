import tensorflow as tf
from tensorflow import keras
from ResNet18 import ResNet18
from keras.datasets import cifar100
import numpy as np

using_gpu_index = 0  # 使用的 GPU 号码
gpu_list = tf.config.experimental.list_physical_devices('GPU')
if len(gpu_list) > 0:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpu_list[using_gpu_index],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
        )
    except RuntimeError as e:
        print(e)
else:
    print("Got no GPUs")

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train = x_train / 255
x_test = x_test / 255

# model construction
model = ResNet18(100)
model.build(input_shape=(None, 32, 32, 3))
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

# train
model.fit(x_train, y_train, epochs=30, batch_size=32, validation_split=0.2)

y_pre = model.predict(x_test)
print(y_pre)
y_pre_label = np.argmax(y_pre, axis=1)
acc = 0
for i in range(len(y_test)):
    if y_pre_label[i] == y_test[i][0]:
        acc+=1
print(acc/len(y_test))