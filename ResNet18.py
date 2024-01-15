from keras import layers, Sequential, Model
from keras.layers import Conv2D, BatchNormalization, ReLU,MaxPool2D,Layer,GlobalAvgPool2D,Dense,Input


class ResNet18(Model):
    def __init__(self, class_num,in_shape, pre_filter_size=7):
        """

        :param in_shape: the shape of each example
        :param class_num: the number of classes
        :param pre_filter_size: the size of filters in the preprocessing layer
        """
        super().__init__()
        # preprocessing layer
        self.pl = Sequential([
            Conv2D(64,(pre_filter_size,pre_filter_size),strides=2,padding='same'),
            BatchNormalization(),
            ReLU(),
            MaxPool2D(pool_size=(3,3),strides=2,padding='same')
        ])

        self.input_layer = Input(in_shape)  # used to reveal output shape

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
        self.avg_pool = GlobalAvgPool2D()
        self.fc = Dense(class_num)

        self.call(self.input_layer)

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


class ResidualBlock(Layer):
    def __init__(self, filters, strides=1):
        super().__init__()
        self.filters = filters
        self.stride = strides

        self.cl1 = Conv2D(filters=filters, kernel_size=(3, 3), strides=strides, padding="same")
        self.bn1 = BatchNormalization()
        self.relu1 = ReLU()

        self.cl2 = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding="same")
        self.bn2 = BatchNormalization()

        if strides != 1:
            self.shortcut = Conv2D(filters=filters, kernel_size=(1, 1), strides=strides)
        else:
            self.shortcut = lambda x: x
        self.relu2 = ReLU()

    def call(self, inputs, *args, **kwargs):
        outputs = self.cl1(inputs)
        outputs = self.bn1(outputs)
        outputs = self.relu1(outputs)

        outputs = self.cl2(outputs)
        outputs = self.bn2(outputs)

        outputs = layers.add([outputs, self.shortcut(inputs)])
        outputs = self.relu2(outputs)

        return outputs
