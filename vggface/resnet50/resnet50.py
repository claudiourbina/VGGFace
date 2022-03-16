import warnings
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input, 
    Conv2D, 
    BatchNormalization, 
    Activation,
    MaxPooling2D,
    AveragePooling2D,
    Flatten,
    Dense,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D
)
from tensorflow.keras.utils import get_file

from .layers import identity_block, conv2d_block

CACHE_DIR = 'models/vggface'
WEIGHTS_PATH = 'https://github.com/claudiourbina/VGGFace/releases/download/v1.0/resnet50.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/claudiourbina/VGGFace/releases/download/v1.0/resnet50_no_top.h5'
LABELS_PATH = 'https://github.com/claudiourbina/VGGFace/releases/download/v1.0/labels.npy'


class ResNet50:
    def __init__(
        self,
        include_top:bool=True,
        input_shape=None,
        pooling:bool=None,
        classes:int=8631
    ) -> None:

        self._include_top = include_top
        self._pooling = pooling
        self._classes = classes
        self._weights = "vggface"
        
        self._input_shape = input_shape
        
        self._bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
        self._model = self._build()

    def _build(
        self
    ):

        inp = Input(shape=self._input_shape)

        # first stage
        x = Conv2D(
            filters=64,
            kernel_size=(7, 7),
            use_bias=False,
            strides=(2, 2),
            padding='same',
            name='conv1'
        )(inp)
        x = BatchNormalization(axis=self._bn_axis, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        
        # second stage
        x = conv2d_block(
            x,
            filters=[64, 64, 256],
            kernel_size=(3, 3),
            use_bias=False,
            strides=(1, 1),
            stage=2,
            block=1
        )
        x = identity_block(
            x,
            filters=[64, 64, 256],
            kernel_size=(3, 3),
            use_bias=False,
            stage=2,
            block=2
        )
        x = identity_block(
            x,
            filters=[64, 64, 256],
            kernel_size=(3, 3),
            use_bias=False,
            stage=2,
            block=3
        )

        # third stage
        x = conv2d_block(
            x,
            filters=[128, 128, 512],
            kernel_size=(3, 3),
            use_bias=False,
            stage=3,
            block=1
        )

        x = identity_block(
            x,
            filters=[128, 128, 512],
            kernel_size=(3, 3),
            use_bias=False,
            stage=3,
            block=2
        )

        x = identity_block(
            x,
            filters=[128, 128, 512],
            kernel_size=(3, 3),
            use_bias=False,
            stage=3,
            block=3
        )

        x = identity_block(
            x,
            filters=[128, 128, 512],
            kernel_size=(3, 3),
            use_bias=False,
            stage=3,
            block=4
        )

        # fourth stage
        x = conv2d_block(
            x,
            filters=[256, 256, 1024],
            kernel_size=(3, 3),
            use_bias=False,
            stage=4,
            block=1
        )

        x = identity_block(
            x,
            filters=[256, 256, 1024],
            kernel_size=(3, 3),
            use_bias=False,
            stage=4,
            block=2
        )

        x = identity_block(
            x,
            filters=[256, 256, 1024],
            kernel_size=(3, 3),
            use_bias=False,
            stage=4,
            block=3
        )

        x = identity_block(
            x,
            filters=[256, 256, 1024],
            kernel_size=(3, 3),
            use_bias=False,
            stage=4,
            block=4
        )

        x = identity_block(
            x,
            filters=[256, 256, 1024],
            kernel_size=(3, 3),
            use_bias=False,
            stage=4,
            block=5
        )

        x = identity_block(
            x,
            filters=[256, 256, 1024],
            kernel_size=(3, 3),
            use_bias=False,
            stage=4,
            block=6
        )
        
        # fifth stage
        x = conv2d_block(
            x,
            filters=[512, 512, 2048],
            kernel_size=(3, 3),
            use_bias=False,
            stage=5,
            block=1
        )

        x = identity_block(
            x,
            filters=[512, 512, 2048],
            kernel_size=(3, 3),
            use_bias=False,
            stage=5,
            block=2
        )

        x = identity_block(
            x,
            filters=[512, 512, 2048],
            kernel_size=(3, 3),
            use_bias=False,
            stage=5,
            block=3
        )

        x = AveragePooling2D(pool_size=(7, 7), name="avg")(x)

        if self._include_top:
            x = Flatten()(x)
            x = Dense(units=self._classes, activation='softmax', name='classifier')(x)
        else:
            if self._pooling == "avg":
                x = GlobalAveragePooling2D()(x)
            elif self._pooling == "max":
                x = GlobalMaxPooling2D()(x)

        model = Model(inputs=inp, outputs=x, name='resnet50')

        if self._include_top:
            weights_path = get_file(
                'resnet50.h5',
                WEIGHTS_PATH,
                cache_subdir=CACHE_DIR
            )
        else:
            weights_path = get_file(
                'resnet50_no_top.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir=CACHE_DIR
            )
        model.load_weights(weights_path)

        return model

    def _preprocessing(
        self,
        x,
        data_format=None
    ):  
        _x = np.copy(x)
        if data_format is None:
            data_format = K.image_data_format()
        assert data_format in {'channels_last', 'channels_first'}

        if data_format == 'channels_first':
            _x = _x[:, ::-1, ...]
            _x[:, 0, :, :] -= 91.4953
            _x[:, 1, :, :] -= 103.8827
            _x[:, 2, :, :] -= 131.0912
        else:
            _x = _x[..., ::-1]
            _x[..., 0] -= 91.4953
            _x[..., 1] -= 103.8827
            _x[..., 2] -= 131.0912

        return _x


    def predict(
        self,
        x,
        postprocessing:bool=False
    ):
        x = self._preprocessing(x)
        y = self._model.predict(x)
        if postprocessing:
            y = self._postprocessing(y)
        return y

    def _postprocessing(
        self,
        y,
        top:int=5
    ):
        if self._include_top:
            if len(y.shape) == 2:
                if y.shape[1] == self._classes:
                    fpath = get_file(
                        'labels.npy',
                        LABELS_PATH,
                        cache_subdir=CACHE_DIR
                    )
                    LABELS = np.load(fpath)
                else:
                    raise ValueError('`decode_predictions` expects '
                                    'a batch of predictions '
                                    '(i.e. a 2D array of shape (samples, 2622)) for V1 or '
                                    '(samples, 8631) for V2.'
                                    'Found array with shape: ' + str(y.shape))
            else:
                raise ValueError('`decode_predictions` expects '
                                'a batch of predictions '
                                '(i.e. a 2D array of shape (samples, 2622)) for V1 or '
                                '(samples, 8631) for V2.'
                                'Found array with shape: ' + str(y.shape))
            results = []
            for pred in y:
                top_indices = pred.argsort()[-top:][::-1]
                result = [[str(LABELS[i].encode('utf8')), pred[i]] for i in top_indices]
                result.sort(key=lambda x: x[1], reverse=True)
                results.append(result)
            return results
        else:
            raise Exception("Can't postprocess without top.")