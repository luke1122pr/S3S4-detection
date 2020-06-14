import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
import os, configparser

import wandb

def allow_gpu_growth():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.compat.v1.Session(config=config))

def set_wandb_config(params, overwrite=False, include_preprocessing_setting=False):
    if include_preprocessing_setting:
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(__file__), '../../config.cfg'))
        params.update({
            'big_exam_bandpass_filter': config['Big_Exam'].getboolean('do_bandpass_filter'),
            'big_exam_bandpass_filter_lowcut': int(config['Big_Exam']['bandpass_filter_lowcut']),
            'big_exam_bandpass_filter_highcut': int(config['Big_Exam']['bandpass_filter_highcut']),

            'audicor_10s_bandpass_filter': config['Audicor_10s'].getboolean('do_bandpass_filter'),
            'audicor_10s_bandpass_filter_lowcut': int(config['Audicor_10s']['bandpass_filter_lowcut']),
            'audicor_10s_bandpass_filter_highcut': int(config['Audicor_10s']['bandpass_filter_highcut']),
        })

    for key, value in params.items():
        if key not in wandb.config._items or overwrite:
            wandb.config.update({key: value})
