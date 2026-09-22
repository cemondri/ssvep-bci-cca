"""
Channel analysis for SSVEP-BCI -- which electrodes carry the SSVEP signal?

Tests each electrode individually and produces a topographic map.
Expected result: occipital dominance (SSVEP originates in visual cortex).

Uses the classification functions from ssvep_cca.py.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import mne

import ssvep_cca as sc   # reuse classify / subject_accuracy

# 64 channel names (from the dataset's 64-channels.loc file, in order)
CHANNEL_NAMES = [
    'Fp1','Fpz','Fp2','AF3','AF4','F7','F5','F3','F1','Fz',
    'F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCz','FC2',
    'FC4','FC6','FT8','T7','C5','C3','C1','Cz','C2','C4',
    'C6','T8','M1','TP7','CP5','CP3','CP1','CPz','CP2','CP4',
    'CP6','TP8','M2','P7','P5','P3','P1','Pz','P2','P4',
    'P6','P8','PO7','PO5','PO3','POz','PO4','PO6','PO8','CB1',
    'O1','Oz','O2','CB2'
]


def per_channel_accuracy(subjects, window_sec=2.0):
    """Accuracy of each single channel, pooled across subjects."""
    scores = []
    for ch in range(64):
        sc.CHANNELS = [ch]                       # use only this channel
        c_sum, t_sum = 0, 0
        for subj in subjects:
            c, t = sc.subject_accuracy(subj, window_sec, 'cca')
            c_sum += c
            t_sum += t
        acc = 100 * c_sum / t_sum
        scores.append(acc)
        print(f"{CHANNEL_NAMES[ch]}: {acc:.1f}%")
    return np.array(scores)


def plot_topomap(channel_scores):
    """Topographic map of per-channel accuracy."""
    # drop channels not in the standard montage
    drop = ['M1', 'M2', 'CB1', 'CB2']
    keep = [i for i, name in enumerate(CHANNEL_NAMES) if name not in drop]
    names = [CHANNEL_NAMES[i] for i in keep]
    scores = channel_scores[keep]

    info = mne.create_info(names, sfreq=250, ch_types='eeg')
    info.set_montage(mne.channels.make_standard_montage('standard_1005'),
                     on_missing='ignore')

    fig, ax = plt.subplots(figsize=(7, 6))
    im, _ = mne.viz.plot_topomap(scores, info, axes=ax, show=False,
                                 cmap='RdYlBu_r', contours=6)
    plt.colorbar(im, ax=ax, label='Accuracy (%)')
    ax.set_title('SSVEP single-channel accuracy\n(occipital dominance)')
    plt.tight_layout()
    plt.savefig('topomap.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    subjects = glob.glob(os.path.join(sc.DATA_DIR, '*.mat'))
    scores = per_channel_accuracy(subjects)

    best = np.argsort(scores)[::-1]
    print("\nTop 10 channels:")
    for i in best[:10]:
        print(f"{CHANNEL_NAMES[i]}: {scores[i]:.1f}%")

    plot_topomap(scores)
