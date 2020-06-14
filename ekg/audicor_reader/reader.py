#!/usr/bin/env python3
import sys
if sys.version_info < (3, 0):
    sys.stdout.write('ERROR: Python2.x is not supported! Please use python3.\n')
    sys.exit(-1)

import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import argparse
import re
import time, datetime
from scipy.signal import spectrogram

try:
    from . import denoise
except:
    import denoise

def generate_spectrogram(raw_data, sampling_rates):
    result = list()
    for signal, sr in zip(raw_data, sampling_rates):
        f, t, Sxx = spectrogram(signal, sr, nperseg=int(sr/20), noverlap=int(sr/80), nfft=2048)
        result.append([f, t, Sxx])
    return result

def save_fig(filename, data, grid=False, peak_indices=None, segment_indices=None, figsize=None):
    mpl.rcParams['agg.path.chunksize'] = 10000
    if figsize is None:
        figsize = (20, 2*data.shape[0])
    fig = plt.figure(figsize=figsize)
    for index_channel, channel_data in enumerate(data):
        fig.add_subplot(data.shape[0], 1, index_channel+1)
        if grid:
            if index_channel < 8: # no voltage grid on heart sounds
                y_major_grid = [510. * i for i in range(-4, 5)]
                for yi in y_major_grid:
                    plt.axhline(yi, linestyle='-', color='r', alpha=0.1)
            x_major_grid = [0.2*1000.*i for i in range(int(10/0.2+1))]
            for xi in x_major_grid:
                plt.axvline(xi, linestyle='-', color='r', alpha=0.1)

        if segment_indices is not None:
            for si in segment_indices:
                plt.axvline(si, linestyle='-', color='black', alpha=1.)
        if peak_indices is not None and index_channel < 8:
            colors = ['g', 'm', 'y', 'k', 'c']
            labels = ['P', 'Q', 'R', 'S', 'T']
            for px, ci, li in zip(peak_indices, colors, labels):
                plt.scatter(px, channel_data[px], color=ci, label=li, alpha=0.5, s=20)
            plt.legend(prop={'size': 6})

        plt.plot(channel_data)
        plt.margins(x=0, y=0)

    fig.tight_layout()
    if filename: fig.savefig(filename)
    else: plt.show()

def save_spectrogram_fig(filename, data, figsize=None):
    mpl.rcParams['agg.path.chunksize'] = 10000
    if figsize is None:
        figsize = (20, 2*len(data))
    fig = plt.figure(figsize=figsize)
    for index_signal, (f, t, Sxx) in enumerate(data):
        ax = fig.add_subplot(len(data), 1, index_signal+1)
        ax.pcolormesh(t, f, Sxx)
        ax.set_ylim(0, 50)
        plt.ylabel('Frequency [Hz]')

    plt.xlabel('Time [sec]')
    fig.tight_layout()
    if filename: fig.savefig(filename)
    else: plt.show()

def get_ekg(filename, do_bandpass_filter=True, filter_lowcut=30, filter_highcut=100):
    with open(filename, 'rb') as f:
        f.seek(0xE8)
        data_length = int.from_bytes(f.read(2), byteorder='little', signed=False)

        f.seek(0xE0)
        number_channels_ekg = int.from_bytes(f.read(2), byteorder='little', signed=False)

        f.seek(0xE4)
        number_channels_hs = int.from_bytes(f.read(2), byteorder='little', signed=False) # heart sound
        number_channels = number_channels_ekg + number_channels_hs

        data = [ list() for _ in range(number_channels) ]

        # data start
        f.seek(0x4B8)
        for index_cycle in range(data_length):
            raw = f.read(2 * number_channels)
            if len(raw) < 2 * number_channels:
                break
            for index_channel in range(number_channels):
                data[index_channel].append(int.from_bytes(
                raw[index_channel*2: (index_channel+1)*2],
                byteorder='little', signed=True))

    data = np.array(data)

    hs_data = data[number_channels_ekg: number_channels_ekg+number_channels_hs]
    if do_bandpass_filter: hs_data = denoise.heart_sound_denoise(hs_data, filter_lowcut, filter_highcut, 1000)
    return data, [1000.]*number_channels # sampling rates

def get_heart_sounds(filename, start_s=0, end_s=np.inf, verbose=True):
    with open(filename, 'rb') as f:
        # reading header
        f.read(0x24) # padding
        number_channels = int.from_bytes(f.read(0x1), byteorder='little')
        while f.read(0x1) != b'\x0F': pass
        main_sampling_rate = float(f.read(0x10)[:0xF].decode('utf-8'))
        channel_sampling_rate = [ float(f.read(0x10)[:0xF].decode('utf-8')) for _ in range(number_channels) ]

        # calculate reading order
        data_cycle = int(main_sampling_rate // channel_sampling_rate[-1])
        index_order = list()
        number_value_per_cycle = [0] * number_channels
        index_value_per_cycle = [list() for _ in range(number_channels)]
        for cycle in range(data_cycle):
            for index_channel in range(number_channels):
                if cycle % (main_sampling_rate // channel_sampling_rate[index_channel]) == 0:
                    index_order.append(index_channel)

        for index_channel, index_value in zip(index_order, range(len(index_order))):
            number_value_per_cycle[index_channel] += 1
            index_value_per_cycle[index_channel].append(index_value)

        # calculate number of cycle
        f.seek(0, 2) # to the end of file
        file_size = f.tell()
        number_cycles = (file_size - 512) // 2 // len(index_order)
        total_time_in_sec = number_cycles * index_order.count(0) / channel_sampling_rate[0]

        if verbose: # print out info
            print('='*37, 'INFO', '='*37)
            print('number of channels:', number_channels)
            print('main sampling rate:', main_sampling_rate)
            for index_channel in range(number_channels):
                print('sampling rate-'+str(index_channel)+':', channel_sampling_rate[index_channel])
            print('channel reading order:', index_order)
            print('total time:', str(datetime.timedelta(seconds=int(total_time_in_sec))))
            print('='*80)
            print('reading... ETA: {:.1f}s'.format(file_size / 1000 / 1000 / 17))

        # reading raw file
        f.seek(0x200) # 512
        values = np.frombuffer(f.read(0x2 * number_cycles * len(index_order)), dtype=np.uint16)
        channel_signals = [ np.ndarray([number_cycles * number_value_per_cycle[i]]) for i in range(number_channels) ]
        for index_channel in range(number_channels):
            for index_value in range(number_value_per_cycle[index_channel]):
                channel_signals[index_channel][index_value::number_value_per_cycle[index_channel]] = values[index_value_per_cycle[index_channel][index_value]::len(index_order)]
        # convert to numpy array
        channel_signals = np.array(channel_signals)

        # cut from start_s to end_s
        end_s = min(end_s, total_time_in_sec)
        if end_s > start_s:
            for index_channel in range(number_channels):
                start_index = int(channel_sampling_rate[index_channel] * start_s)
                end_index = int(channel_sampling_rate[index_channel] * end_s)
                channel_signals[index_channel] = channel_signals[index_channel][start_index:end_index]

        return channel_signals, channel_sampling_rate

def convert_time_to_sec(time_string='0:0:0'):
    x = time.strptime(time_string,'%H:%M:%S')
    return datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds()

def generate_filenames(parse_args):
    raw_data_filename = os.path.basename(parse_args.filename)
    if parse_args.start_time or parse_args.end_time:
        raw_data_filename += '.'+ (parse_args.start_time if parse_args.start_time else '0:0:0')
        raw_data_filename += '-'+ (parse_args.end_time if parse_args.end_time else '23:59:59')
    if parse_args.do_denoise:
        raw_data_filename += '_denoised'
    spectrogram_filename = raw_data_filename + '.spectrogram.png'
    raw_data_filename += '.png'

    return raw_data_filename, spectrogram_filename

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Produce ekg and heart_sound figure.')
    parser.add_argument('filename', help='Filename to read. Must be *.bin or *.raw (case-insensitive).')
    parser.add_argument(
                '-sx',
                '--size-x',
                help='X-axis size of saved figure. (default: 20)',
                dest='size_x',
                default=20)

    parser.add_argument(
                '-sy',
                '--size-y',
                help='Y-axis size of saved figure. (default: 20)',
                dest='size_y',
                default=20)

    parser.add_argument(
                '-st',
                '--start-time',
                help='Start time of plt. Only works with *.raw. (default: 0:0:0)',
                dest='start_time')

    parser.add_argument(
                '-et',
                '--end-time',
                help='End time of plt. Only works with *.raw. (default: 23:59:59)',
                dest='end_time')

    parser.add_argument(
                '-fsg',
                '--force-spectrogram',
                help='Calculate spectrogram of which has the data length longer than 60s.',
                dest='force_spectrogram',
                action='store_true'
                )

    parser.add_argument(
                '-dn',
                '--denoise',
                help='Apply wavelet thresholding and high-pass filter to ekg for denoising.',
                dest='do_denoise',
                action='store_true'
                )

    parser.add_argument(
                '-sg',
                '--segment',
                help='Apply segmentation to ekg.',
                dest='do_segment',
                action='store_true'
                )

    args = parser.parse_args()

    # generate filenames
    raw_data_filename, spectrogram_filename = generate_filenames(args)
    print('Save to {} & {}!'.format(raw_data_filename, spectrogram_filename))

    figsize = (int(args.size_x), int(args.size_y))
    if re.search('.*.bin', args.filename, re.IGNORECASE): # EKG
        peak_indices, segment_indices = None, None
        ekg_raw, sampling_rates = get_ekg(args.filename)
        if args.do_denoise:
            ekg_raw = denoise.denoise(ekg_raw, number_channels=8) # NOTE: fixed channel number
        if args.do_segment:
            import ecgseg
            ekg_signal = ekg_raw if args.do_denoise else denoise.denoise(ekg_raw, number_channels=8)
            peak_indices, segment_indices = ecgseg.predict('./2000-0.75.h5', ekg_signal)

        ekg_spectrograms = generate_spectrogram(ekg_raw, sampling_rates)
        save_fig(raw_data_filename, ekg_raw, grid=True, peak_indices=peak_indices, segment_indices=segment_indices, figsize=figsize)
        save_spectrogram_fig(spectrogram_filename, ekg_spectrograms, figsize=figsize)

    elif re.search('.*.raw', args.filename, re.IGNORECASE): # Heart Sound
        if args.do_segment:
            print('''--segment option is ignored, since it's specified for EKGs.''')

        start_s = convert_time_to_sec(args.start_time) if args.start_time else 0
        end_s = convert_time_to_sec(args.end_time) if args.end_time else np.inf

        heart_sounds, sampling_rates = get_heart_sounds(args.filename, start_s, end_s)

        if args.do_denoise: # NOTE: this may only work with 6-channel .raw file
            if heart_sounds.shape[0] != 6:
                print('Warning: --denoise option may only work with 6-channel .raw files, while {:d}-channel signal is given!'.format(heart_sounds.shape[0]))

            heart_sounds[0:1] = denoise.heart_sound_denoise(heart_sounds[0:1], 30, 100, sampling_rates[0])# heart sound
            heart_sounds[1] = denoise.denoise(heart_sounds[1][np.newaxis, ...], number_channels=1)[0] # EKG

        save_fig(raw_data_filename, heart_sounds, figsize=figsize)

        if end_s - start_s < 60 or args.force_spectrogram:
            heart_sounds_spectrograms = generate_spectrogram(heart_sounds, sampling_rates)
            save_spectrogram_fig(spectrogram_filename, heart_sounds_spectrograms, figsize=figsize)
        else:
            print('''The signal is too long, skipping the spectrogram! Use -fsg to bypass the check!''')

    else:
        print('ERROR: filename must be *.bin or *.raw.')
