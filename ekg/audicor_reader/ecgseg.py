import numpy as np
from keras.models import load_model
import pickle
import reader

# use CPU only
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def normalize(X, means_and_stds=None):
    if means_and_stds is None:
        means = [ X[..., i].mean(dtype=np.float32) for i in range(X.shape[-1]) ]
        stds = [ X[..., i].std(dtype=np.float32) for i in range(X.shape[-1]) ]
    else:
        means = means_and_stds[0]
        stds = means_and_stds[1]

    normalized_X = np.zeros_like(X, dtype=np.float32)
    for i in range(X.shape[-1]):
        normalized_X[..., i] = X[..., i].astype(np.float32) - means[i]
        normalized_X[..., i] = normalized_X[..., i] / stds[i]
    return normalized_X, (means, stds)

def get_segment(probabilities): # (10000, 6)
    diff = np.swapaxes(np.diff(probabilities, axis=0), 0, 1) # (6, 10000-1) p,q,r,s,t
    probabilities = np.swapaxes(probabilities, 0, 1) # (6, 10000)
    peak_indices = [[] for _ in range(5)]

    for index_peak in range(5): # p,q,r,s,t
        peak_threshold = probabilities[index_peak].mean() + 3 * probabilities[index_peak].std()

        local_peaks = np.logical_and( (diff[index_peak]>=0)[:-1], (diff[index_peak]<=0)[1:] ) # (10000 - 2)
        local_peaks = np.logical_and( local_peaks, (probabilities[index_peak] > peak_threshold)[1:-1] )

        for location_peak in np.where(local_peaks)[0] + 1:
            local_maximum = probabilities[index_peak, max(0, location_peak-300):min(9999, location_peak+300)].max()
            if probabilities[index_peak, location_peak] == local_maximum:
                peak_indices[index_peak].append(location_peak+1)

    P, Q, R, S, T = 0, 1, 2, 3, 4
    # T start
    segment_indices = list()
    if peak_indices[P][0] > peak_indices[T][0]:
        for i in range(len(peak_indices[P])):
            segment_indices.append( (peak_indices[P][i] + peak_indices[T][i])//2 )
    else: # P start
        for i in range(len(peak_indices[P])-1):
            segment_indices.append( (peak_indices[P][i+1] + peak_indices[T][i])//2 )

    return peak_indices, segment_indices

def predict(model_filename, ekg_signal):
    # load model
    model = load_model(model_filename, compile=False)

    # normalize signal
    with open('./seg_means_and_stds.pickle', 'rb') as f:
        means_and_stds = pickle.load(f)
    ekg_signal = np.swapaxes(ekg_signal[:8], 0, 1) # (8, 10000) -> (10000, 8)
    ekg_signal = ekg_signal.reshape(1, 10000, 8)
    ekg_signal, _ = normalize(ekg_signal, means_and_stds)

    # do prediction
    probabilities = model.predict(ekg_signal)[0]
    return get_segment(probabilities) # post processing

if __name__ == '__main__':
    ekg, sampling_rates = reader.get_ekg('/home/toosyou/ext_ssd/Cardiology/交大-normal/大檢查audicor/NOR059/PP-01_001852.bin')
    # ekg, sampling_rates = reader.get_ekg('/home/toosyou/ext_ssd/Cardiology/交大-normal/大檢查audicor/NOR014/PP-01_000914.bin')
    print(predict('./2000-0.75.h5', ekg))
