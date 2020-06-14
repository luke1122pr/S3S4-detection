from tensorflow import keras
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalAveragePooling2D, Reshape, Dense, Permute, multiply
import tensorflow.keras.backend as K

class LeftCropLike(keras.layers.Layer):
    def call(self, inputs, **kwargs):
        source, target = inputs
        source_shape = K.shape(source)
        target_shape = K.shape(target)
        return source[:, source_shape[1]-target_shape[1]: ]

    def compute_output_shape(self, input_shape):
        return input_shape[1]

class CenterCropLike(keras.layers.Layer):
    def call(self, inputs, **kwargs):
        source, target = inputs
        source_shape = K.shape(source)
        target_shape = K.shape(target)
        length_diff = source_shape[1]-target_shape[1]
        return source[:, length_diff//2 : length_diff//2+target_shape[1] ]

    def compute_output_shape(self, input_shape):
        return input_shape[1]

class PaddingLike(keras.layers.Layer):
    def call(self, inputs, **kwargs):
        source, target = inputs
        source_shape = K.shape(source)
        target_shape = K.shape(target)
        length_diff = target_shape[1]-source_shape[1]
        return K.temporal_padding(source, padding=(length_diff//2, length_diff - length_diff//2))

    def compute_output_shape(self, input_shape):
        return input_shape[1]

def squeeze_excite_block_1d(tensor, ratio=16):
    init = tensor
    filters = init._shape_val[-1]
    se_shape = (1, filters)

    se = GlobalAveragePooling1D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = multiply([init, se])
    return x

def squeeze_excite_block_2d(tensor, ratio=16):
    init = tensor
    filters = init._shape_val[-1]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = multiply([init, se])
    return x