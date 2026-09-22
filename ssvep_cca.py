"""
SSVEP-BCI classification using CCA and FBCCA.

Detects which flickering frequency a user is looking at, from EEG,
on the Tsinghua Benchmark Dataset (35 subjects, 40 targets, 64 channels).

Methods:
  - CCA   (Canonical Correlation Analysis) -- the standard SSVEP method
  - FBCCA (Filter Bank CCA) -- decomposes EEG into harmonic sub-bands

Neither method requires training (zero-calibration): reference signals
are generated mathematically, so every trial is classified independently.

Dataset is NOT included (large .mat files). Download from:
  http://bci.med.tsinghua.edu.cn/download.html
"""

import os
import glob
import numpy as np
import scipy.io
from sklearn.cross_decomposition import CCA
from scipy.signal import butter, filtfilt

# --- Configuration ---
DATA_DIR = 'data'                 # folder containing S1.mat ... S35.mat
FS = 250                          # sampling rate (Hz)
START = 160                       # analysis window start (stimulus onset + ~140ms delay)

# Occipital channels (visual cortex), 0-based indices:
# Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2
CHANNELS = [47, 53, 54, 55, 56, 57, 60, 61, 62]

# Candidate targets (well-separated, easy to distinguish)
TARGET_INDICES = [0, 2, 4, 7]     # positions in the dataset
TARGET_FREQS   = [8, 10, 12, 15]  # corresponding frequencies (Hz)


# ============================================================
#  Reference signals (shared by CCA and FBCCA)
# ============================================================
def make_reference(freq, n_samples, fs, n_harmonics=3):
    """Sine/cosine reference for a frequency, including harmonics."""
    t = np.arange(n_samples) / fs
    refs = []
    for h in range(1, n_harmonics + 1):
        refs.append(np.sin(2 * np.pi * h * freq * t))
        refs.append(np.cos(2 * np.pi * h * freq * t))
    return np.array(refs)


def cca_score(trial, reference):
    """Canonical correlation between an EEG trial and a reference."""
    X = trial.T                   # (time, channels)
    Y = reference.T               # (time, reference components)
    cca = CCA(n_components=1)
    cca.fit(X, Y)
    X_c, Y_c = cca.transform(X, Y)
    return np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]


# ============================================================
#  FBCCA -- filter bank
# ============================================================
def bandpass(signal, low, high, fs):
    """Band-pass filter (4th-order Butterworth, zero-phase)."""
    nyq = fs / 2
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal, axis=-1)


def fbcca_score(trial, freq, n_samples, fs, n_bands=5):
    """FBCCA: split EEG into harmonic sub-bands, CCA each, weighted sum.

    Sub-band n covers [n*8, 88] Hz. Lower bands (fundamental) get
    higher weight, following Chen et al. (2015).
    """
    reference = make_reference(freq, n_samples, fs)
    total = 0.0
    for n in range(1, n_bands + 1):
        low = n * 8               # 8, 16, 24, 32, 40 Hz
        high = 88
        if low >= high:
            continue
        band = bandpass(trial, low, high, fs)
        r = cca_score(band, reference)
        weight = n ** (-1.25) + 0.25
        total += weight * (r ** 2)
    return total


# ============================================================
#  Classification
# ============================================================
def classify(trial, n_samples, fs, method='cca'):
    """Assign a trial to the candidate frequency with the highest score."""
    scores = []
    for f in TARGET_FREQS:
        if method == 'cca':
            scores.append(cca_score(trial, make_reference(f, n_samples, fs)))
        else:  # fbcca
            scores.append(fbcca_score(trial, f, n_samples, fs))
    return TARGET_FREQS[int(np.argmax(scores))]


def subject_accuracy(mat_path, window_sec, method='cca'):
    """Correct / total trials for one subject at a given window length."""
    data = scipy.io.loadmat(mat_path)['data']   # (64, 1500, 40, 6)
    n_samples = int(window_sec * FS)
    end = START + n_samples

    correct, total = 0, 0
    for i, target in enumerate(TARGET_INDICES):
        true_freq = TARGET_FREQS[i]
        for block in range(6):
            trial = data[CHANNELS, START:end, target, block]
            pred = classify(trial, n_samples, FS, method)
            if pred == true_freq:
                correct += 1
            total += 1
    return correct, total


def group_curve(subjects, windows, method='cca'):
    """Pooled accuracy across subjects, for each window length."""
    results = []
    for w in windows:
        c_sum, t_sum = 0, 0
        for subj in subjects:
            c, t = subject_accuracy(subj, w, method)
            c_sum += c
            t_sum += t
        acc = 100 * c_sum / t_sum
        results.append(acc)
        print(f"  {round(w, 2)} s: {acc:.1f}%  ({method})")
    return results


# ============================================================
#  Main -- CCA vs FBCCA comparison
# ============================================================
if __name__ == '__main__':
    subjects = glob.glob(os.path.join(DATA_DIR, '*.mat'))
    print(f"Found {len(subjects)} subjects")

    windows = [0.5, 1, 2, 4]

    print("=== CCA ===")
    cca_curve = group_curve(subjects, windows, method='cca')
    print("=== FBCCA ===")
    fbcca_curve = group_curve(subjects, windows, method='fbcca')

    print("\nWindows:", windows)
    print("CCA:  ", [round(x, 1) for x in cca_curve])
    print("FBCCA:", [round(x, 1) for x in fbcca_curve])
