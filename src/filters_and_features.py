import numpy as np
from scipy.signal import butter, lfilter
import librosa

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def apply_filters(signal, fs):
    # High-pass filter at 200 Hz
    b_hp, a_hp = butter_highpass(cutoff=200, fs=fs, order=5)
    filtered_signal = lfilter(b_hp, a_hp, signal)
    return filtered_signal

def compute_features(frame):
    N = len(frame)
    epsilon = 1e-5

    # Zero Crossing Rate (ZCR)
    sgn = np.sign(frame)
    sgn[sgn == 0] = 1  # Treat zero as positive
    ZCR = (1 / (2 * N)) * np.sum(np.abs(sgn[1:] - sgn[:-1]))

    # Log-Energy
    energy = np.sum(frame ** 2) / N
    log_energy = 10 * np.log10(epsilon + energy)

    # Normalized Autocorrelation Coefficient (C1)
    numerator = np.sum(frame[1:] * frame[:-1])
    denominator = np.sum(frame ** 2)
    C1 = numerator / (epsilon + denominator)

    # LPC Analysis to get First Predictor Coefficient and Prediction Error
    p = 12  # LPC order for 8 kHz sampling rate

    try:
        a = librosa.lpc(frame, order=p)
        first_predictor_coeff = a[1]  # Skip a[0], which is always 1
        # Compute prediction error
        predicted = lfilter([0] + -1 * a[1:].tolist(), 1, frame)
        error = frame - predicted
        error_energy = np.sum(error ** 2) / N
        # Normalized Prediction Error (Ep)
        Ep = log_energy - 10 * np.log10(epsilon + error_energy)
    except Exception:
        # If LPC fails, set default values
        first_predictor_coeff = 0.0
        Ep = 0.0

    # Assemble features into a vector
    feature_vector = np.array([ZCR, log_energy, C1, first_predictor_coeff, Ep])

    return feature_vector

def extract_features_labels(audio, labels, sample_rate, frame_length):
    frame_size = int(frame_length * sample_rate)  # At 8kHz -> 80 samples in each frame
    num_frames = (len(audio) - frame_size) // frame_size + 1
    features_list = []
    labels_list = []

    for i in range(num_frames):
        start_idx = i * frame_size
        end_idx = start_idx + frame_size
        frame = audio[start_idx:end_idx]
        frame_label = labels[start_idx:end_idx]
        # Majority vote label for the frame
        label = int(np.round(np.mean(frame_label)))
        features = compute_features(frame)
        features_list.append(features)
        labels_list.append(label)

    features_array = np.vstack(features_list)
    labels_array = np.array(labels_list)
    return features_array, labels_array
