import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Layer,
    Conv2D,
    BatchNormalization,
    Activation
)

def conv2d_block(
    inp, 
    filters:list,
    kernel_size:tuple,  
    stage:int, 
    block:int,
    strides:tuple=(2, 2), 
    use_bias:bool=False
):
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    name = f"conv{stage}_{block}"

    x = Conv2D(
        filters=filters[0], 
        kernel_size=(1, 1), 
        strides=strides, 
        use_bias=use_bias,
        name=f"{name}_red"
    )(inp)
    x = BatchNormalization(
        axis=bn_axis, 
        name=f"{name}_bn1"
    )(x)
    x = Activation('relu')(x)

    x = Conv2D(
        filters=filters[1],
        kernel_size=kernel_size, 
        padding='same', 
        use_bias=use_bias,
        name=f"{name}"
    )(x)
    x = BatchNormalization(
        axis=bn_axis, 
        name=f"{name}_bn2"
    )(x)
    x = Activation('relu')(x)

    x = Conv2D(
        filters=filters[2],
        kernel_size=(1, 1), 
        name=f"{name}_inc",
        use_bias=use_bias
    )(x)
    x = BatchNormalization(
        axis=bn_axis, 
        name=f"{name}_bn3"
    )(x)

    short = Conv2D(
        filters=filters[2],
        kernel_size=(1, 1),
        strides=strides, 
        use_bias=use_bias,
        name=f"{name}_short",
    )(inp)
    short = BatchNormalization(
        axis=bn_axis, 
        name=f"{name}_bn4"
    )(short)

    x = tf.keras.layers.add([x, short])
    x = Activation('relu')(x)
    return x

def identity_block(
    inp, 
    filters:list, 
    kernel_size:tuple,  
    stage:int, 
    block:int,
    use_bias:bool=False):

    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    name = f"conv{stage}_{block}"

    x = Conv2D(
        filters=filters[0], 
        kernel_size=(1, 1), 
        use_bias=use_bias, 
        name=f"{name}_red"
    )(inp)
    x = BatchNormalization(
        axis=bn_axis, 
        name=f"{name}_bn1"
    )(x)
    x = Activation('relu')(x)

    x = Conv2D(
        filters=filters[1], 
        kernel_size=kernel_size, 
        use_bias=use_bias,
        padding='same', 
        name=f"{name}"
    )(x)
    x = BatchNormalization(
        axis=bn_axis, 
        name=f"{name}_bn2"
    )(x)
    x = Activation('relu')(x)

    x = Conv2D(
        filters=filters[2], 
        kernel_size=(1, 1), 
        use_bias=use_bias, 
        name=f"{name}_inc"
    )(x)
    x = BatchNormalization(
        axis=bn_axis, 
        name=f"{name}_bn3"
    )(x)

    x = tf.keras.layers.add([x, inp])
    x = Activation('relu')(x)
    return x