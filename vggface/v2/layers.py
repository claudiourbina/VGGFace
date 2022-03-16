
class Conv2DBlock(Layer):
    def __init__(
        self,
        filters:list,
        kernel_size:tuple,
        stage:int,
        block:int,
        strides:tuple=(2,2),
        use_bias:bool=False
    ):
        super(Conv2DBlock, self).__init__()
        self._bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
        self._name = f"conv{stage}_{block}"
        
        # layers
        self.conv1 = Conv2D(
            filters=filters[0],
            kernel_size=(1,1),
            strides=strides,
            use_bias=use_bias,
            name=f"{self._name}_red"
        )

        self.bn1 = BatchNormalization(
            axis=self._bn_axis,
            name=f"{self._name}_bn1"
        )

        self.act1 = Activation('relu')

        self.conv2 = Conv2D(
            filters=filters[1],
            kernel_size=kernel_size,
            padding='same',
            use_bias=use_bias,
            name=f"{self._name}"
        )

        self.bn2 = BatchNormalization(
            axis=self._bn_axis,
            name=f"{self._name}_bn2"
        )

        self.act2 = Activation('relu')

        self.conv3 = Conv2D(
            filters=filters[2],
            kernel_size=(1,1),
            use_bias=use_bias,
            name=f"{self._name}_inc"
        )

        self.bn3 = BatchNormalization(
            axis=self._bn_axis,
            name=f"{self._name}_bn3"
        )

        self.act3 = Activation('relu')

        self.short = Conv2D(
            filters=filters[2],
            kernel_size=(1,1),
            strides=strides,
            use_bias=use_bias,
            name=f"{self._name}_short"
        )

        self.bn4 = BatchNormalization(
            axis=self._bn_axis,
            name=f"{self._name}_bn3"
        )

    def call(
        self,
        x,
        training=False
    ):
        x1 = self.conv1(x)
        x1 = self.bn1(x1, training=training)
        x1 = self.act1(x1)

        x1 = self.conv2(x1)
        x1 = self.bn2(x1, training=training)
        x1 = self.act2(x1)

        x1 = self.conv3(x1)
        x1 = self.bn3(x1, training=training)

        x2 = self.short(x)
        x2 = self.bn4(x2, training=training)

        x3 = tf.keras.layers.add([x1, x2])
        x3 = self.act3(x3)

        return x3


class IdentityBlock(Layer):
    def __init__(
        self,
        filters:list,
        kernel_size:tuple,
        stage:int,
        block:int,
        use_bias:bool=False
    ):
        super(IdentityBlock, self).__init__()
        self._bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
        self._name = f"conv{stage}_{block}"

        # layers
        self.conv1 = Conv2D(
            filters=filters[0],
            kernel_size=(1,1),
            use_bias=use_bias,
            name=f"{self._name}_red"
        )

        self.bn1 = BatchNormalization(
            axis=self._bn_axis,
            name=f"{self._name}_bn1"
        )

        self.act1 = Activation('relu')

        self.conv2 = Conv2D(
            filters=filters[1],
            kernel_size=kernel_size,
            padding='same',
            use_bias=use_bias,
            name=f"{self._name}"
        )

        self.bn2 = BatchNormalization(
            axis=self._bn_axis,
            name=f"{self._name}_bn2"
        )

        self.act2 = Activation('relu')

        self.conv3 = Conv2D(
            filters=filters[2],
            kernel_size=(1,1),
            use_bias=use_bias,
            name=f"{self._name}_inc"
        )

        self.bn3 = BatchNormalization(
            axis=self._bn_axis,
            name=f"{self._name}_bn3"
        )

        self.act3 = Activation('relu')

    def call(
        self,
        x,
        training:bool=False
    ):
        x1 = self.conv1(x)
        x1 = self.bn1(x1, training=training)
        x1 = self.act1(x1)

        x1 = self.conv2(x1)
        x1 = self.bn2(x1, training=training)
        x1 = self.act2(x1)

        x1 = self.conv3(x1)
        x1 = self.bn3(x1, training=training)

        x2 = tf.keras.layers.add([x1, x])
        x2 = self.act3(x2)

        return x2