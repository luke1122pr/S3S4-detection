import numpy as np
from tqdm import tqdm

import re
import pandas as pd
import datetime
import warnings

import better_exceptions; better_exceptions.hook()

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from ekg.audicor_reader import denoise
from ekg.audicor_reader import reader

import multiprocessing as mp
from functools import partial

import configparser
config = configparser.ConfigParser()
config.read('./config.cfg')

ABNORMAL_DIRS = config['Big_Exam']['abnormal_dirs'].split(', ')
NORMAL_DIRS = config['Big_Exam']['normal_dirs'].split(', ')
LABEL_FILENAME = config['Big_Exam']['label_filename']
DIRTYDATA_FILENAME = config['Big_Exam']['dirtydata_filename']

OUTPUT_DIR = config['Big_Exam']['output_dir']
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_PROCESSES = mp.cpu_count()*2

def ekg_denoise(X):
    '''Denoise the EKG signal
        X: (?, 10, 10000)
    '''
    warnings.filterwarnings("ignore")
    for i, xi in enumerate(tqdm(X, desc='ekg denoise')):
        X[i] = denoise.ekg_denoise(xi, number_channels=8) # only denoise the first 8 ekg channels
    return

def get_all_filenames(base_dirnames, extension='.bin', remove_prefix=True):
    '''Return a list that contains all the filenames under the base_dirname with the extension.
    '''
    filenames = list()
    for dirname in base_dirnames:
        for directory, sub_directory, filelist in os.walk(dirname, topdown=True):
            for fn in filelist:
                if fn.endswith(extension):
                    # remove the base directory name
                    full_filename = os.path.join(directory, fn)
                    filenames.append(full_filename[len(dirname): ] if remove_prefix and full_filename.startswith(dirname) else full_filename)
    return filenames

def path_is_dirty(pi, dirtydata_df):
    for dp in dirtydata_df['dirty']:
        if dp == dp and dp in pi: return True # dirty
    for dp in dirtydata_df['gray_zone']: # not-so-dirty data
        if dp == dp and dp in pi: return True # dirty
    return False # clean

def generate_survival_data(event_names=None):
    def get_info(path_string):
        path_parts = path_string.split('/')

        patient_code = path_parts[1]
        patient_measuring_date = None
        if re.match(r'(^(A|a|H|c|V|[0-9]))', path_parts[2]):
            patient_measuring_date = re.findall(r"[0-9]{3,}", path_parts[2])[0]

        return patient_code, patient_measuring_date

    def drop_dirty(df):
        dirtydata_df = pd.read_excel(DIRTYDATA_FILENAME)
        filter_function = lambda row: not path_is_dirty(row['filename'], dirtydata_df)
        df = df[df.apply(filter_function, axis=1)].reset_index(drop=True)
        return df

    def append_survival_status(df, event_names):
        for event_name in event_names:
            event_dur_name = event_name + '_dur'

            # 1: event occurred, 0: survived, -1: pass event, ignore
            df[event_name + '_censoring_status'] = 1 # event occurred
            df.loc[df[event_dur_name] != df[event_dur_name], event_name + '_censoring_status'] = 0 # Nan
            df.loc[(df.follow_dur <= 0) | (df[event_dur_name] < 0) | (df.filename.str.contains('NG')), event_name + '_censoring_status'] = -1

            df[event_name + '_survival_time'] = df[event_dur_name] # event occurred
            df.loc[df[event_dur_name] != df[event_dur_name], event_name + '_survival_time'] = df.follow_dur[df[event_dur_name] != df[event_dur_name]] # Nan
            df.loc[(df.follow_dur <= 0) | (df[event_dur_name] < 0) | (df.filename.str.contains('NG')), event_name + '_survival_time'] = 0

        return df

    if event_names is None:
        event_names = ['ADHF', 'MI', 'Stroke', 'CVD', 'Mortality']

    # load the label files and fix the nan issue
    ahf2017_df = pd.read_excel(LABEL_FILENAME, skiprows=1).replace('#NULL!', np.nan)

    abnormal_filenames = get_all_filenames(ABNORMAL_DIRS)

    # create empty dataframe for output
    output_df = pd.DataFrame(index=np.arange(len(abnormal_filenames)), 
                            columns=['filename', 'code', 'follow_dur'] + [en + '_dur' for en in event_names])
    output_df['filename'] = abnormal_filenames

    for i, filename in enumerate(abnormal_filenames):
        code, measuring_date = get_info(filename)
        ahf2017_df_row = ahf2017_df[ahf2017_df.code == code]

        if ahf2017_df_row.empty: # drop the file if there's no label for it
            continue

        if measuring_date is None:
            follow_dur = int(ahf2017_df_row.follow_dur)
        else:
            follow_dur = int(((ahf2017_df_row.follow_date) - datetime.datetime.strptime(measuring_date, '%m%d%y')).dt.days)

        for event_name in event_names:
            try:
                if measuring_date is None: # don't get measuring date in path, find it in outcome.csv
                    dur = int(ahf2017_df_row[event_name + '_dur'])
                else:
                    dur = int(((ahf2017_df_row[event_name + '_date']) - datetime.datetime.strptime(measuring_date, '%m%d%y')).dt.days)
            except: # TODO: find the exact except
                dur = np.nan

            output_df.loc[output_df.index[i], event_name+'_dur'] = dur

        output_df.loc[output_df.index[i], 'code'] = code
        output_df.loc[output_df.index[i], 'follow_dur'] = follow_dur
        
    # drop files with no label
    output_df = output_df.dropna(subset=['code']).reset_index(drop=True)

    # rename code column to subject_id
    output_df = output_df.rename(columns={'code': 'subject_id'}, errors='ignore')

    # remove dirty data
    output_df = drop_dirty(output_df)

    # convert durations to survival_time and censoring_status
    output_df = append_survival_status(output_df, event_names)

    # remove duration data
    output_df = output_df.drop(columns=['follow_dur'] + [en + '_dur' for en in event_names])

    return output_df

def mp_get_ekg(filenames, do_bandpass_filter=True, filter_lowcut=30, filter_highcut=100, desc='Loading data'):
    X = list()
    __mp_get_ekg = partial(reader.get_ekg, do_bandpass_filter=do_bandpass_filter, filter_lowcut=filter_lowcut, filter_highcut=filter_highcut)
    with mp.Pool(processes=NUM_PROCESSES) as workers:
        for xi, _ in tqdm(workers.imap(__mp_get_ekg, filenames), total=len(filenames), desc=desc):
            X.append(xi)

    return np.array(X)

def load_normal_data(do_bandpass_filter=True, filter_lowcut=30, filter_highcut=100):
    filenames = get_all_filenames(NORMAL_DIRS, remove_prefix=False)
    return mp_get_ekg(filenames, do_bandpass_filter, filter_lowcut, filter_highcut, 'Loading normal data')

def load_abnormal_data(filenames_df, do_bandpass_filter=True, filter_lowcut=30, filter_highcut=100):
    full_filenames = list(map(lambda fn: ABNORMAL_DIRS[0]+fn, filenames_df.filename))
    return mp_get_ekg(full_filenames, do_bandpass_filter, filter_lowcut, filter_highcut, 'Loading abnormal data')

if __name__ == '__main__':
    # parse setting
    do_bandpass_filter = config['Big_Exam'].getboolean('do_bandpass_filter')
    filter_lowcut = int(config['Big_Exam']['bandpass_filter_lowcut'])
    filter_highcut = int(config['Big_Exam']['bandpass_filter_highcut'])

    abnormal_event_df = generate_survival_data()
    abnormal_event_df.to_csv(os.path.join(OUTPUT_DIR, 'abnormal_event.csv') , index=False)

    normal_X = load_normal_data(do_bandpass_filter, filter_lowcut, filter_highcut)
    ekg_denoise(normal_X)
    np.save(os.path.join(OUTPUT_DIR, 'normal_X.npy'), normal_X)

    abnormal_X = load_abnormal_data(abnormal_event_df, do_bandpass_filter, filter_lowcut, filter_highcut)
    ekg_denoise(abnormal_X)
    np.save(os.path.join(OUTPUT_DIR, 'abnormal_X.npy'), abnormal_X)
