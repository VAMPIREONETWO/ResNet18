import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential, Model


class ResNet18(Model):
    def __init__(self, class_num):
        super().__init__()
        # preprocessing layer
        self.pl = Sequential([
            layers.Conv2D(64,(7,7),strides=2,padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size=(3,3),strides=2,padding='same')
        ])

        # Residual Blocks
        self.block1 = ResidualBlock(64)
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(128, 2)
        self.block4 = ResidualBlock(128)
        self.block5 = ResidualBlock(256, 2)
        self.block6 = ResidualBlock(256)
        self.block7 = ResidualBlock(512, 2)
        self.block8 = ResidualBlock(512)

        # Fully Connected Layer
        # self.fc = Sequential(
        #     layers.GlobalAveragePooling2D(),
        #     layers.Dense(class_num)
        # )
        self.avg_pool = layers.GlobalAvgPool2D()
        self.fc = layers.Dense(class_num)

    def call(self, inputs, training=None, mask=None):
        outputs = self.pl(inputs)
        outputs = self.block1(outputs)
        outputs = self.block2(outputs)
        outputs = self.block3(outputs)
        outputs = self.block4(outputs)
        outputs = self.block5(outputs)
        outputs = self.block6(outputs)
        outputs = self.block7(outputs)
        outputs = self.block8(outputs)
        outputs = self.avg_pool(outputs)
        outputs = self.fc(outputs)
        return outputs


class ResidualBlock(layers.Layer):
    def __init__(self, filters, strides=1):
        super(ResidualBlock, self).__init__()
        self.filters = filters
        self.stride = strides

        self.cl1 = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=strides, padding="same")
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.Activation("relu")

        self.cl2 = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding="same")
        self.bn2 = layers.BatchNormalization()

        if strides != 1:
            self.shortcut = layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=strides)
        else:
            self.shortcut = lambda x: x
        self.relu2 = layers.Activation("relu")

    def call(self, inputs, *args, **kwargs):
        outputs = self.cl1(inputs)
        outputs = self.bn1(outputs)
        outputs = self.relu1(outputs)

        outputs = self.cl2(outputs)
        outputs = self.bn2(outputs)

        outputs = layers.add([outputs, self.shortcut(inputs)])
        outputs = self.relu2(outputs)

        return outputs
