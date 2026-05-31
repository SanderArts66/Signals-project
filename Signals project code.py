import numpy as np
import matplotlib.pyplot as plt # plotting library 
import librosa # library that analyses audio



AI_FILES = ["Ai1.wav", "Ai2.mp3", "Ai3.mp3", "Ai4.wav", "Ai5.wav", "Ai6.wav", "Ai7.wav"]

HUMAN_FILES = ["Human1.mp3", "Human2.mp3", "Human3.mp3", "Human4.mp3", "Human5.mp3",
               "Human6.mp3", "Human7.mp3",]

DURATION_SEC = 5      # seconds of audio to analyse per file, if audio is longer than 10 min,
# a FFT_size should be implemented here and in function compute_fft. However, since the
#application does not need more than 10 min to determine AI or human, this was left out. 

ROLLOFF_PERC = 0.85   # spectral rolloff threshold, could be changed, however since we are
# comparing Ai relative to Human, the exact value matters little.



# Purpose : Reads one audio file from disk, removes silence, and trims it to
#           a fixed length so every sample is comparable. If there is no audio, 
#           an error is raised.
# Input   : filepath     — path to the audio file
#           duration_sec — how many seconds to keep 
# Output  : signal — 1-D numpy array of normalised amplitude values
#           sr     — sample rate in Hz
def load_audio(filepath, duration_sec):
    signal, sample_rate = librosa.load(filepath, sr=None, mono=True)
    signal, _ = librosa.effects.trim(signal, top_db=20)

    N = int(duration_sec * sample_rate)
    if len(signal) < N:
        signal = np.pad(signal, (0, N - len(signal)))  
    else:
        signal = signal[:N]

    if np.max(np.abs(signal)) == 0:
        raise ValueError(f"Audio file appears to be silent: {filepath}")

    signal = signal / (np.max(np.abs(signal)) + 1e-10)
    return signal, sample_rate

# Purpose : Converts a time-domain signal into the frequency domain using the
#           Discrete Fourier Transform. A Hann window is applied first to
#           prevent spectral leakage at the edges of the signal. Only the positive signals
#           are evaluated. 
# Input   : signal — 1-D array of amplitude values (from load_audio)
#           sr     — sample rate in Hz (from load_audio)
#           n_fft  — FFT size; if None the full signal length is used
# Output  : freqs — 1-D array of frequency values in Hz for each FFT bin
#           mag   — 1-D array of magnitudes (how strong each frequency is)

def compute_fft(signal, sample_rate):
    N = len(signal)
    window     = np.hanning(N)
    dft        = np.fft.fft(signal * window, n=N)
    magnitude  = np.abs(dft[: N // 2])
    freqs      = np.fft.fftfreq(N, d=1 / sample_rate)[: N // 2]
    return freqs, magnitude

# Purpose : Computes all 5 features from the FFT output for exploration.
#           No classifier is applied here — the goal is just to see the numbers.
# Input   : signal — 1-D array of amplitude values (from load_audio)
#           sr     — sample rate in Hz
#           freqs  — 1-D array of frequency values in Hz (from compute_fft)
#           mag    — 1-D array of FFT magnitudes (from compute_fft)
# Output  : dictionary with 5 float values, one per feature

def extract_features( freqs, mag):
    magnitude_safe  = mag + 1e-10
    geometric_mean  = np.exp(np.mean(np.log(magnitude_safe)))
    arithmetic_mean = np.mean(magnitude_safe)
    spectral_flat   = geometric_mean / arithmetic_mean

    spectral_centroid = np.sum(freqs * mag) / (np.sum(mag))

    hf_ratio      = np.sum(mag[freqs > 4000]) / (np.sum(mag))

    cumsum        = np.cumsum(mag)
    rolloff_idx   = min(np.searchsorted(cumsum, ROLLOFF_PERC * cumsum[-1]), len(freqs) - 1)
    spec_rolloff  = freqs[rolloff_idx]

    spec_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * mag) / np.sum(mag))

    return {
        "spectral_flatness": spectral_flat,
        "spectral_centroid": spectral_centroid,
        "high_freq_ratio":   hf_ratio,
        "spectral_rolloff":  spec_rolloff,
        "spectral_bandwidth": spec_bandwidth}

# Purpose : Runs the full pipeline (load → FFT → features) on every file in a
#           list. No classification is performed — just feature extraction.
#           Stores signal and FFT data so they can be plotted later.
# Input   : file_list — list of filename stems
#           label     — "AI" or "Human", stored for reference in the plots
# Output  : list of result dictionaries, one per file, each containing:
#             name, label, signal, sr, freqs, mag, feats

def process_files(file_list, label):
    results = []
    for filepath in file_list:
        name = filepath.split(".")[0]
        try:
            signal, sample_rate = load_audio(filepath, DURATION_SEC)
            freqs, magnitude = compute_fft(signal, sample_rate)
            features      = extract_features(freqs, magnitude)
            results.append({
                "name":   name,
                "label":  label,
                "signal": signal,
                "sample_rate":     sample_rate,
                "freqs":  freqs,
                "magnitude":    magnitude,
                "features":  features})
            print(f"  [OK] {name}")
        except Exception as e:
            print(f"  [ERR] {filepath}: {e}")
    return results

def plot_waveforms_and_ffts(results, group_label, chunk_size=4):
    for start in range(0, len(results), chunk_size):
        chunk = results[start:start + chunk_size]
        n = len(chunk)
        fig, axes = plt.subplots(n, 2, figsize=(14, 3 * n))
        fig.suptitle(f"{group_label} — Time Signals & FFT Spectra ({start+1}-{start+n})", fontsize=12)
        for i, r in enumerate(chunk):
            time = np.arange(len(r["signal"])) / r["sample_rate"]
            axes[i, 0].plot(time, r["signal"], linewidth=0.5)
            axes[i, 0].set_title(r["name"])
            axes[i, 0].set_ylabel("Amplitude")
            axes[i, 0].set_xlabel("Time [s]")
            axes[i, 0].grid(alpha=0.4)
            axes[i, 1].plot(r["freqs"], r["magnitude"], linewidth=0.5)
            axes[i, 1].set_title(r["name"])
            axes[i, 1].set_ylabel("Magnitude")
            axes[i, 1].set_xlabel("Frequency [Hz]")
            axes[i, 1].set_xlim(0, 8000)
            axes[i, 1].grid(alpha=0.4)
        plt.tight_layout()

def plot_feature_distributions(ai_results, human_results):
    feature_names = list(ai_results[0]["features"].keys())
    fig, axes = plt.subplots(1, len(feature_names), figsize=(4 * len(feature_names), 5))
    fig.suptitle("Feature Distributions: AI vs Human", fontsize=12)
    for ax, feat in zip(axes, feature_names):
        ai_vals    = [r["features"][feat] for r in ai_results]
        human_vals = [r["features"][feat] for r in human_results]
        bp = ax.boxplot([ai_vals, human_vals], labels=["AI", "Human"], patch_artist=True)
        bp["boxes"][0].set_facecolor("#4a90d9")
        bp["boxes"][1].set_facecolor("#e87040")
        ax.set_title(feat.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()

def plot_feature_table(ai_results, human_results):
    for group, results in [("AI", ai_results), ("Human", human_results)]:
        feature_names = list(results[0]["features"].keys())
        col_labels = ["Sample"] + [f.replace("_", " ").title() for f in feature_names]
        rows = [[r["name"]] + [f"{r['features'][f]:.4f}" for f in feature_names] for r in results]
        fig, ax = plt.subplots(figsize=(max(12, 2 * len(col_labels)), 0.45 * len(rows) + 2))
        ax.axis("off")
        fig.suptitle(f"Feature Values — {group}", fontsize=12)
        table = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        for j in range(len(col_labels)):
            table[0, j].set_facecolor("#8b0000")
            table[0, j].set_text_props(color="white", fontweight="bold")
        plt.tight_layout()


if __name__ == "__main__":

    print("Loading AI files…")
    ai_results    = process_files(AI_FILES,    label="AI")

    print("\nLoading Human files…")
    human_results = process_files(HUMAN_FILES, label="Human")

    if not ai_results or not human_results:
        print("\nNo results — check your file paths.")
    else:
        print("\nGenerating plots…")

        # Per-sample waveforms + FFTs
        plot_waveforms_and_ffts(ai_results,    "AI Voices")
        plot_waveforms_and_ffts(human_results, "Human Voices")

        # Feature distribution box plots
        plot_feature_distributions(ai_results, human_results)

        # Feature value table
        plot_feature_table(ai_results, human_results)

        plt.show()