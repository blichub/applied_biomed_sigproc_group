"""
Week 1 Signal Processing Utilities and Documentation

---
ECG Signals:
- MIT-BIH Arrhythmia Database segments, multiple cardiac rhythms
- Features: P, QRS, T waves; RR intervals; QRS duration; RR variability; P/T wave morphology
- Filtering: Power line (60 Hz), baseline wander; FIR/IIR filters (see scipy.signal)

Acceleration & PPG Signals:
- Activities: Lying, standing, sitting, walking, running
- Features: Acceleration axes, PPG waveform, heart rate
- Spectrogram and power spectral density analysis

ECG & PPG Comparison:
- Cardiac beat detection, rhythm differences
- IBI (interbeat interval) analysis

---
Key Questions:
- What are the main aspects of each cardiac rhythm in terms of ECG and interbeat intervals?
- Which features of the signals would be interesting to distinguish the different rhythms?
- Can you reduce ECG interferences with a digital filter without distorting the signal?
- What are the main differences in acceleration and PPG signals between activities?
- When is cardiac activity most visible? Can you always see it?
- Which signal (ECG or PPG) is best to detect cardiac beats? Why?
"""

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import pathlib
import pickle

# --- ECG Data Loading ---
def load_ecg_data(data_file=None):
    """Load ECG signals and metadata from .npz file."""
    if data_file is None:
        data_file = pathlib.Path("data/ecg_data.npz")
    with np.load(data_file) as data:
        ecg = data["signals"]
        fs = data["fs"].item()
        leads = data["leads"]
        rhythms = data["rhythms"]
        beats = data["beats"]
    beats = [indices[np.isfinite(indices)].astype("int64") for indices in beats]
    return ecg, fs, leads, rhythms, beats

# --- Activity Data Loading ---
def load_activity_data(data_file=None):
    """Load acceleration, PPG, and heart rate from .npz file."""
    if data_file is None:
        data_file = pathlib.Path("data/activity_data.npz")
    with np.load(data_file) as data:
        time = data["time"]
        acceleration = data["acceleration"]
        ppg = data["ppg"]
        hr_time = data["hr_time"]
        hr = data["hr"]
    return time, acceleration, ppg, hr_time, hr

# --- ECG+PPG Data Loading ---
def load_ecg_ppg_data(data_file=None):
    """Load ECG+PPG signals from .pkl file."""
    if data_file is None:
        data_file = pathlib.Path("data/ecg_ppg_data.pkl")
    with open(data_file, mode="rb") as f:
        return pickle.load(f)

# --- ECG Plotting ---
def plot_ecg(time, ecg, beats, leads, rhythm):
    fig, axes = plt.subplots(
        ecg.shape[0] + 1, 1, sharex="all", squeeze=False, constrained_layout=True
    )
    fig.suptitle(f"Rhythm: {rhythm}")
    for i in range(ecg.shape[0]):
        ax = axes.flat[i]
        ax.plot(time, ecg[i], linewidth=1)
        ax.plot(time[beats], ecg[i, beats], ".")
        ax.grid()
        ax.set_ylabel(leads[i])
    ax = axes.flat[-1]
    ax.plot(time[beats[1:]], np.diff(time[beats]), ".-", linewidth=1)
    ax.grid()
    ax.set_ylim(0.0, 3.0)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("IBI [s]")

# --- Activity Plotting ---
def plot_signals(time, acceleration, ppg, hr_time, hr, start=None, end=None, title=None):
    mask = np.ones(time.size, dtype="bool")
    hr_mask = np.ones(hr_time.size, dtype="bool")
    if start is not None:
        mask = np.logical_and(mask, time >= start)
        hr_mask = np.logical_and(hr_mask, hr_time >= start)
    if end is not None:
        mask = np.logical_and(mask, time <= end)
        hr_mask = np.logical_and(hr_mask, hr_time <= end)
    time = time[mask]
    acceleration = acceleration[mask]
    ppg = ppg[mask]
    hr_time = hr_time[hr_mask]
    hr = hr[hr_mask]

    fig, axes = plt.subplots(3, 1, sharex="all", constrained_layout=True)
    if title is not None:
        plt.suptitle(title)
    plt.sca(axes.flat[0])
    plt.plot(time, acceleration, linewidth=1)
    plt.grid()
    plt.ylabel("Acceleration [g]")
    plt.legend(["X-axis", "Y-axis", "Z-axis"], loc="upper right")
    plt.sca(axes.flat[1])
    plt.plot(time, ppg, linewidth=1)
    plt.grid()
    plt.ylabel("PPG")
    plt.sca(axes.flat[2])
    plt.plot(hr_time, hr, linewidth=1)
    plt.ylim(0.0, 240.0)
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Heart rate [bpm]")

# --- Spectrogram Plotting ---
def plot_spectrogram(time, ppg, hr_time, hr):
    fs = 1.0 / np.median(np.diff(time))
    plt.figure(constrained_layout=True)
    plt.specgram(ppg, Fs=fs, NFFT=512, detrend="mean")
    plt.plot(hr_time, hr / 60.0, color="white")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")

# --- Power Spectral Density ---
def plot_power_spectral_density(time, ppg, hr_time, hr, start=None, end=None, title=None):
    fs = 1.0 / np.median(np.diff(time))
    mask = np.ones(time.size, dtype="bool")
    hr_mask = np.ones(hr_time.size, dtype="bool")
    if start is not None:
        mask = np.logical_and(mask, time >= start)
        hr_mask = np.logical_and(hr_mask, hr_time >= start)
    if end is not None:
        mask = np.logical_and(mask, time <= end)
        hr_mask = np.logical_and(hr_mask, hr_time <= end)
    f, s = scipy.signal.welch(ppg[mask], fs=fs, nperseg=256, nfft=1024)
    mean_hr = np.mean(hr[hr_mask])

    plt.figure(constrained_layout=True)
    if title is not None:
        plt.suptitle(title)
    plt.plot(f, s, linewidth=1)
    plt.axvline(mean_hr / 60.0, color="tab:red", label="Mean heart rate")
    plt.grid()
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power spectral density")
    plt.legend(loc="upper right")

# --- ECG+PPG Plotting ---
def plot_ecg_ppg_signal(signal):
    fig, axes = plt.subplots(3, 1, sharex="all", constrained_layout=True)
    fig.suptitle(f"Rhythm: {signal['rhythm']}")

    def plot_ecg(ax):
        t = signal["ecg"]["time"]
        x = signal["ecg"]["signal"]
        beats = signal["ecg"]["beats"]
        ax.plot(t, x, linewidth=1)
        ax.plot(beats, np.interp(beats, t, x), ".")
        ax.grid()
        ax.set_ylabel("ECG")

    def plot_ppg(ax):
        t = signal["ppg"]["time"]
        x = signal["ppg"]["signal"]
        beats = signal["ppg"]["beats"]
        ax.plot(t, x, linewidth=1)
        ax.plot(beats, np.interp(beats, t, x), ".")
        ax.grid()
        ax.set_ylabel("PPG")

    def plot_ibi(ax):
        ecg_beats = signal["ecg"]["beats"]
        ppg_beats = signal["ppg"]["beats"]
        ax.plot(ecg_beats[1:], np.diff(ecg_beats), ".-", linewidth=1, label="ECG")
        ax.plot(ppg_beats[1:], np.diff(ppg_beats), ".-", linewidth=1, label="PPG")
        ax.grid()
        ax.set_ylim(0.0, 3.0)
        ax.set_ylabel("IBI [s]")
        ax.legend(loc="upper right")

    plot_ecg(axes.flat[0])
    plot_ppg(axes.flat[1])
    plot_ibi(axes.flat[2])
    axes.flat[-1].set_xlabel("Time [s]")

# --- Filtering Utilities ---
def get_lead_v1(ecg):
    """Return lead V1 from ECG array."""
    return ecg[0, 1]

def bandstop_filter(signal, fs, freq=60, bandwidth=2):
    """Apply a bandstop filter to remove powerline interference."""
    from scipy.signal import iirdesign, filtfilt
    wp = [freq - bandwidth/2, freq + bandwidth/2]
    ws = [freq - bandwidth, freq + bandwidth]
    gpass = 1
    gstop = 40
    b, a = iirdesign(wp=[w/fs*2 for w in wp], ws=[w/fs*2 for w in ws], gpass=gpass, gstop=gstop, ftype='butter')
    return filtfilt(b, a, signal)
