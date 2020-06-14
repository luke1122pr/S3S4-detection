import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, AveragePooling2D, Dense, Add, Concatenate, Reshape
from tensorflow.keras.layers import GaussianNoise

from ..layers import LeftCropLike, CenterCropLike
from ..layers import squeeze_excite_block_1d, squeeze_excite_block_2d
from ..layers.sincnet import SincConv1D
from ..layers.non_local import non_local_block
import numpy as np
def _ekg_branch(input_data, nlayers, nfilters, kernel_length, kernel_initializer, se_block, skip_connection):
    ekg = input_data
    for i in range(nlayers):
        shortcut = ekg
        ekg = Conv1D(nfilters, kernel_length, activation='relu', padding='same',
                        kernel_initializer=kernel_initializer, name='ekg_branch_conv_{}'.format(i))(ekg)
        ekg = BatchNormalization(name='ekg_branch_bn_{}'.format(i))(ekg)
        if se_block: ekg = squeeze_excite_block_1d(ekg)

        if i != 0 and skip_connection:
            ekg = Add(name='ekg_branch_skip_merge_{}'.format(i))([ekg, shortcut])

        ekg = MaxPooling1D(2, padding='same', name='ekg_branch_maxpool_{}'.format(i))(ekg)

    return ekg

def _heart_sound_branch(input_data, sincconv_filter_length, sincconv_nfilters, hs_nfilters, sampling_rate, nlayers, kernel_length, kernel_initializer, se_block, skip_connection, name_prefix=''):
    hs = input_data
    sincconv_filter_length = sincconv_filter_length - (sincconv_filter_length+1) % 2
    hs = SincConv1D(sincconv_nfilters, sincconv_filter_length, sampling_rate, name='{}sincconv'.format(name_prefix))(hs)
    hs = BatchNormalization(name='{}bn_0'.format(name_prefix))(hs)

    for i in range(nlayers):
        shortcut = hs
        hs = Conv1D(hs_nfilters, kernel_length, activation='relu', padding='same',
                        kernel_initializer=kernel_initializer, name='{}conv_{}'.format(name_prefix, i+1))(hs)
        hs = BatchNormalization(name='{}bn_{}'.format(name_prefix, i+1))(hs)
        if se_block: hs = squeeze_excite_block_1d(hs)

        if i != 0 and skip_connection:
            hs = Add(name='{}skip_merge_{}'.format(name_prefix, i+1))([hs, shortcut])

        hs = MaxPooling1D(2, padding='same', name='{}maxpool_{}'.format(name_prefix, i+1))(hs)

    return hs

def _wavelet_hs_branch(input_data, scale_length, nfilters, kernel_initializer, nlayers, se_block, skip_connection):
    hs = input_data
    hs = Conv2D(nfilters, (scale_length, 1), activation='relu', padding='same', 
                    kernel_initializer=kernel_initializer, name='wavelet_hs_conv_0')(hs)
    hs = BatchNormalization(name='wavelet_hs_bn_0')(hs)
    hs = AveragePooling2D((2, 1), padding='same', name='wavelet_hs_avgpool_0')(hs) # (?, /2, 10000)

    hs = Conv2D(nfilters, (scale_length//2, 1), activation='relu', padding='same', 
                    kernel_initializer=kernel_initializer, name='wavelet_hs_conv_1')(hs)
    hs = BatchNormalization(name='wavelet_hs_bn_1')(hs)
    hs = AveragePooling2D((2, 1), padding='same', name='wavelet_hs_avgpool_1')(hs) # (?, /4, 10000)

    for i in range(nlayers):
        shortcut = hs
        hs = Conv2D(nfilters, 3, activation='relu', padding='same',
                        kernel_initializer=kernel_initializer, name='wavelet_hs_conv_{}'.format(i+2))(hs)
        hs = BatchNormalization(name='wavelet_hs_bn_{}'.format(i+2))(hs)
        if se_block: hs = squeeze_excite_block_2d(hs)

        if i != 0 and skip_connection:
            hs = Add(name='wavelet_skip_merge_{}'.format(i+2))([hs, shortcut])

        hs = MaxPooling2D((2, 2), padding='same', name='wavelet_maxpool_{}'.format(i+2))(hs)

    hs = Lambda(lambda hs: K.max(hs, axis=1))(hs)
    return hs

def backbone(config, include_top=False, classification=True, classes=2):
    # inputs
    inputs = list()
    if config.wavelet:
        if config.n_ekg_channels != 0:
            ekg_input = Input((config.sampling_rate*10, config.n_ekg_channels), name='ekg_input')
            inputs.append(ekg_input)

        if config.n_hs_channels != 0:
            heart_sound_input = Input((config.wavelet_scale_length, config.sampling_rate*10, config.n_hs_channels), name='hs_input')
            inputs.append(heart_sound_input)
    else:
        ekg_hs_input = Input((config.sampling_rate*10, config.n_ekg_channels + config.n_hs_channels), name='ekg_hs_input')
        inputs.append(ekg_hs_input)

        if config.n_ekg_channels != 0:
            ekg_input = Lambda(lambda x, n_ekg_channels: x[:, :, :n_ekg_channels], 
                                    arguments={'n_ekg_channels': config.n_ekg_channels}, 
                                    name='ekg_input')(ekg_hs_input) # (10000, 8)
        if config.n_hs_channels != 0:
            heart_sound_input = Lambda(lambda x, n_hs_channels: x[:, :, -n_hs_channels:], 
                                    arguments={'n_hs_channels': config.n_hs_channels}, 
                                    name='hs_input')(ekg_hs_input) # (10000, 2)

    if config.include_info:
        info_input = Input((len(config.infos), ), name='info_input') # sex, age, height, weight, BMI

        info = info_input
        if config.info_apply_noise:
            infos = list()
            for i, info_string in enumerate(config.infos):
                this_info = Lambda(lambda x, i: x[:, i], arguments={'i': i}, name='{}_input'.format(info_string))(info)
                this_info = Reshape((1, ))(this_info)
                infos.append(GaussianNoise(stddev=config.info_norm_noise_std[i])(this_info))
            info = Concatenate(axis=-1, name='noise_info_input')(infos)

        inputs.append(info_input)
    print('?',np.array(inputs).shape)
    # ekg branch
    if config.n_ekg_channels != 0:
        ekg = _ekg_branch(ekg_input, 
                            config.branch_nlayers,
                            config.ekg_nfilters,
                            config.ekg_kernel_length, 
                            config.kernel_initializer,
                            config.se_block, 
                            config.skip_connection)

    # heart sound branch
    if config.n_hs_channels != 0:
        if config.wavelet:
            hs = _wavelet_hs_branch(heart_sound_input, 
                                    config.wavelet_scale_length, 
                                    config.hs_nfilters,
                                    config.kernel_initializer, 
                                    config.branch_nlayers, 
                                    config.se_block, 
                                    config.skip_connection)
        else:
            hs_outputs = list()
            for i in range(config.n_hs_channels):
                hs = Lambda(lambda x, i: K.expand_dims(x[:, :, i], -1), 
                                        arguments={'i': i}, 
                                        name='hs_split_{}'.format(i))(heart_sound_input)
                hs_outputs.append(_heart_sound_branch(hs, config.sincconv_filter_length,
                                                            config.sincconv_nfilters, 
                                                            config.hs_nfilters,
                                                            config.sampling_rate,
                                                            config.branch_nlayers,
                                                            config.hs_kernel_length, config.kernel_initializer,
                                                            config.se_block,
                                                            config.skip_connection, name_prefix='hs_branch_{}_'.format(i)))
            if config.n_hs_channels >= 2:
                hs = Add(name='hs_merge')(hs_outputs)
            else: # no need to merge
                hs = hs_outputs[0]

    # merge block
    if config.n_ekg_channels != 0 and config.n_hs_channels != 0:
        if config.crop_center:
            ekg = CenterCropLike(name='ekg_crop')([ekg, hs])
        else:
            ekg = LeftCropLike(name='ekg_crop')([ekg, hs])
        output = Concatenate(axis=-1, name='hs_ekg_merge')([hs, ekg])
    else:
        output = ekg if config.n_ekg_channels != 0 else hs

    if include_top: # final layers
        for i in range(config.final_nlayers):
            shortcut = output
            output = Conv1D(config.final_nfilters, config.final_kernel_length, activation='relu', padding='same',
                                kernel_initializer=config.kernel_initializer, name='final_conv_{}'.format(i))(output)
            output = BatchNormalization(name='final_bn_{}'.format(i))(output)
            if config.se_block: output = squeeze_excite_block_1d(output)

            if i != 0 and config.skip_connection:
                output = Add(name='final_skip_merge_{}'.format(i))([output, shortcut])

            if i >= config.final_nlayers - config.final_nonlocal_nlayers: # the final 'final_nonlocal_nlayers' layers
                output = non_local_block(output, compression=2, mode='embedded')

            if i != config.final_nlayers-1 or config.prediction_head: # not the final output
                output = MaxPooling1D(2, padding='same', name='final_maxpool_{}'.format(i))(output)
        
        # prediction head setup
        if hasattr(config, 'prediction_head') and config.prediction_head:
            outputs = list()

            for i_class in range(classes):
                head_output = output
                for i in range(config.prediction_nlayers):
                    shortcut = head_output
                    head_output = Conv1D(config.prediction_nfilters, config.prediction_kernel_length, activation='relu', padding='same',
                                kernel_initializer=config.kernel_initializer, name='pred_{}_conv_{}'.format(i_class, i))(head_output)
                    head_output = BatchNormalization(name='pred_{}_bn_{}'.format(i_class, i))(head_output)
                    if config.se_block: head_output = squeeze_excite_block_1d(head_output)

                    if i != 0 and config.skip_connection:
                        head_output = Add(name='pred_{}_skip_{}'.format(i_class, i))([head_output, shortcut])

                    if i != config.prediction_nlayers - 1: # not the last layer
                        head_output = MaxPooling1D(2, padding='same', name='pred_{}_maxpool_{}'.format(i_class, i))(head_output)

                head_output = GlobalAveragePooling1D(name='pred_{}_gap'.format(i_class))(head_output)

                # info MLP
                if config.include_info:
                    head_output = Concatenate(axis=-1, name='pred_{}_info_concat'.format(i_class))([head_output, info])
                    for j in range(config.info_nlayers):
                        head_output = Dense(config.info_units, activation='relu', name='pred_{}_info_dense_{}'.format(i_class, j))(head_output)

                head_output = Dense(1, activation='linear', name='pred_{}_output'.format(i_class))(head_output)
                outputs.append(head_output)
            
            if len(outputs) > 1:
                output = Concatenate(axis=-1, name='output')(outputs)
            else:
                output = Lambda(lambda x: x, name='output')(outputs[0])
        else:
            output = GlobalAveragePooling1D(name='output_gap')(output)
            # info MLP
            if config.include_info:
                output = Concatenate(axis=-1, name='final_info_concat')([output, info])
                for i in range(config.info_nlayers):
                    output = Dense(config.info_units, name='final_info_dense_{}'.format(i))(output)

            output = Dense(classes, activation='softmax' if classification else 'linear', name='output')(output) # classification or regression

    return Model(inputs, [output])