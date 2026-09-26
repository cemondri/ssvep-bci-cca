# SSVEP-BCI Frequency Classification

This project uses CCA and FBCCA methods to identify which frequency a user is looking at, using the Tsinghua Benchmark dataset.

## Overview

SSVEP (Steady-State Visual Evoked Potential) is a natural brain response that happens when a person looks at a flickering light — the brain begins to oscillate at the same frequency as the stimulus. In this project, I built an analysis pipeline that detects exactly which frequency a user is focusing on. It uses a "zero-calibration" approach, which means the system works right away without needing to train a machine learning model for each new person.

## Dataset

I used the Tsinghua Benchmark dataset for this analysis. It includes EEG recordings from 35 subjects who looked at 40 different target frequencies. To save space, the actual data files are not included in this project, but you can download them directly from the official source:
http://bci.med.tsinghua.edu.cn/download.html

## Methods

I applied two mathematical methods to analyze the signals:

- **CCA (Canonical Correlation Analysis):** the standard, traditional method used to match brain waves with target frequencies.
- **FBCCA (Filter Bank CCA):** a more advanced version that splits the EEG signal into harmonic sub-bands, using the harmonic structure of SSVEP more effectively — especially at short time windows.

## Results

Five main findings from the analysis:

- **High accuracy:** The classification works well, reaching about 97% accuracy with a 4-second window.
- **Speed–accuracy trade-off:** There is a clear S-curve between window length and accuracy. The optimal window is around 1–2 seconds. For a BCI this matters, since users want fast responses.
- **Method comparison:** FBCCA performs noticeably better than standard CCA, especially at short time windows — exactly where a fast BCI needs it.
- **Occipital dominance confirmed:** The most informative electrodes are all over the visual cortex at the back of the head. A single occipital electrode (Oz) alone reached around 80% accuracy in every subject — suggesting a single-channel setup could be practical for wearable BCI.
- **Experience effect:** I tested whether subjects with prior BCI experience performed better. There was a small trend in that direction, but it was not statistically significant (p = 0.48), possibly due to the small and unbalanced groups (8 experienced vs 27 naive). I include this as an honest negative result.

## Honesty & Limitations

This project does not invent a new algorithm; it is a practical application of the existing CCA and FBCCA methods. The candidate set was limited to four well-separated frequencies, and the experience-effect comparison was underpowered. The main point is to demonstrate honest methodology and clear analysis, rather than to present a perfect classifier.

## Files

- `ssvep_cca.py` — CCA and FBCCA classification, speed–accuracy comparison
- `channel_analysis.py` — per-channel accuracy and topographic map
- `requirements.txt` — dependencies

## How to run

​```bash
pip install -r requirements.txt
# place the .mat files in a folder named data/
python ssvep_cca.py
python channel_analysis.py
​```
