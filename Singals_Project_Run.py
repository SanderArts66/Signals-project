import numpy as np
import matplotlib.pyplot as plt
import librosa
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG — change this to the file you want to test (no extension needed)
# ─────────────────────────────────────────────────────────────────────────────

INPUT_FILE = "AI1"    # ← put the filename you want to test here

DURATION_SEC       = 5     # seconds of audio to analyse
FFT_SIZE           = None  # None = use full signal length
FLATNESS_THRESHOLD = 0.1   # above this → AI, below → Human
ZCR_THRESHOLD      = 0.05  # above this → AI, below → Human

# ─────────────────────────────────────────────────────────────────────────────

# Purpose : Given a filename stem (e.g. "AI1"), searches the current directory
#           for a matching audio file with any common extension.
# Input   : stem — filename without extension
# Output  : full filename string if found, otherwise raises FileNotFoundError
def resolve_path(stem):
    for ext in [".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"]:
        p = Path(stem + ext)
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f"No audio file found for '{stem}' — tried .mp3 .wav .flac .ogg .m4a .aac"
    )

# Purpose : Reads one audio file from disk, removes silence, and trims it to
#           a fixed length.
# Input   : filepath     — path to the audio file
#           duration_sec — how many seconds to keep
# Output  : signal — 1-D numpy array of normalised amplitude values
#           sr     — sample rate in Hz
def load_audio(filepath, duration_sec=5):
    signal, sr = librosa.load(filepath, sr=None, mono=True)
    signal = signal / (np.max(np.abs(signal)) + 1e-10)
    signal, _ = librosa.effects.trim(signal, top_db=20)
    N = int(duration_sec * sr)
    signal = signal[:N]
    signal = signal / (np.max(np.abs(signal)) + 1e-10)
    return signal, sr

# Purpose : Converts the time-domain signal into the frequency domain using
#           the DFT. A Hann window is applied to prevent spectral leakage.
# Input   : signal — 1-D array of amplitude values
#           sr     — sample rate in Hz
#           n_fft  — FFT size; None = use full signal length
# Output  : freqs — frequency values in Hz for each bin
#           mag   — magnitude (strength) of each frequency
def compute_fft(signal, sr, n_fft=None):
    N = len(signal)
    n = n_fft if n_fft else N
    window = np.hanning(N)
    dft    = np.fft.fft(signal * window, n=n)
    mag    = np.abs(dft[: n // 2])
    freqs  = np.fft.fftfreq(n, d=1 / sr)[: n // 2]
    return freqs, mag

# Purpose : Computes the two features used for classification.
#           Spectral flatness captures how noise-like the spectrum is.
#           ZCR captures the fine texture of the time-domain signal.
# Input   : signal — 1-D array of amplitude values
#           sr     — sample rate in Hz
#           freqs  — frequency array from compute_fft
#           mag    — magnitude array from compute_fft
# Output  : dictionary with two float values: spectral_flatness and zcr
def extract_features(signal, sr, freqs, mag):
    # Spectral flatness: geometric mean / arithmetic mean of the spectrum
    mag_safe      = mag + 1e-10
    geo_mean      = np.exp(np.mean(np.log(mag_safe)))
    arith_mean    = np.mean(mag_safe)
    spectral_flat = geo_mean / arith_mean

    # Zero-crossing rate: how often the signal crosses zero per second
    zcr = np.mean(librosa.feature.zero_crossing_rate(signal)[0])

    return {
        "spectral_flatness": spectral_flat,
        "zcr":               zcr,
    }

# Purpose : Classifies the sample as AI or Human based on the two features.
#           Both features must agree for a confident prediction; if they
#           disagree the result is marked as "Uncertain".
# Input   : features — the dictionary returned by extract_features
# Output  : a string: "AI", "Human", or "Uncertain"
def classify(features):
    flat_says_ai = features["spectral_flatness"] > FLATNESS_THRESHOLD
    zcr_says_ai  = features["zcr"]               > ZCR_THRESHOLD

    if flat_says_ai and zcr_says_ai:
        return "AI"
    elif not flat_says_ai and not zcr_says_ai:
        return "Human"
    else:
        return "Uncertain"

# Purpose : Shows the time-domain waveform and FFT spectrum side by side for
#           the single input file.
# Input   : signal    — 1-D array of amplitude values
#           sr        — sample rate in Hz
#           freqs     — frequency array from compute_fft
#           mag       — magnitude array from compute_fft
#           name      — filename stem used as the plot title
#           prediction — classification result string
# Output  : none — displays a matplotlib figure
def plot_signal(signal, sr, freqs, mag, name, prediction):
    color = "#e84040" if prediction == "AI" else \
            "#40a060" if prediction == "Human" else "#e8a040"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{name}  —  Prediction: {prediction}", fontsize=14, color=color)

    # Time domain
    time = np.arange(len(signal)) / sr
    ax1.plot(time, signal, linewidth=0.5, color=color)
    ax1.set_title("Time Signal")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Amplitude")
    ax1.grid(alpha=0.4)

    # Frequency domain
    ax2.plot(freqs, mag, linewidth=0.5, color=color)
    ax2.set_title("FFT Spectrum")
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("Magnitude")
    ax2.set_xlim(0, 8000)
    ax2.grid(alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

# Purpose : Prints a small results table to the console showing the two
#           feature values, their thresholds, and the final prediction.
# Input   : name       — filename stem
#           features   — dictionary from extract_features
#           prediction — classification result string
# Output  : none — prints to console
def print_results(name, features, prediction):
    print("\n" + "=" * 45)
    print(f"  File       : {name}")
    print(f"  Prediction : {prediction}")
    print("-" * 45)
    print(f"  {'Feature':<22} {'Value':>8}   {'Threshold':>9}")
    print(f"  {'-'*22} {'-'*8}   {'-'*9}")
    print(f"  {'Spectral Flatness':<22} {features['spectral_flatness']:>8.5f}   {FLATNESS_THRESHOLD:>9.5f}")
    print(f"  {'Zero-Crossing Rate':<22} {features['zcr']:>8.5f}   {ZCR_THRESHOLD:>9.5f}")
    print("=" * 45 + "\n")


if __name__ == "__main__":

    name = Path(INPUT_FILE).stem

    try:
        signal, sr = load_audio(resolve_path(INPUT_FILE), DURATION_SEC)
        freqs, mag = compute_fft(signal, sr, FFT_SIZE)
        features   = extract_features(signal, sr, freqs, mag)
        prediction = classify(features)

        print_results(name, features, prediction)
        plot_signal(signal, sr, freqs, mag, name, prediction)
        plt.show()

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")