import numpy as np
import matplotlib.pyplot as plt
import librosa
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG — edit these lists with your actual file names (no extension needed)
# ─────────────────────────────────────────────────────────────────────────────

AI_FILES = [
    "AI1", "AI2", "AI3", "AI4", "AI5", "AI6", "AI7",
]

HUMAN_FILES = [
    "Human1", "Human2", "Human3", "Human4", "Human5", "Human6", "Human7",
]

DURATION_SEC = 5      # seconds of audio to analyse per file
FFT_SIZE     = None   # None = use full signal length
ROLLOFF_PERC = 0.85   # spectral rolloff threshold (85%)

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
#           a fixed length so every sample is comparable.
# Input   : filepath     — path to the audio file
#           duration_sec — how many seconds to keep (default: 5)
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

# Purpose : Converts a time-domain signal into the frequency domain using the
#           Discrete Fourier Transform. A Hann window is applied first to
#           prevent spectral leakage at the edges of the signal.
# Input   : signal — 1-D array of amplitude values (from load_audio)
#           sr     — sample rate in Hz (from load_audio)
#           n_fft  — FFT size; if None the full signal length is used
# Output  : freqs — 1-D array of frequency values in Hz for each FFT bin
#           mag   — 1-D array of magnitudes (how strong each frequency is)
def compute_fft(signal, sr, n_fft=None):
    N = len(signal)
    n = n_fft if n_fft else N
    window = np.hanning(N)
    dft    = np.fft.fft(signal * window, n=n)
    mag    = np.abs(dft[: n // 2])
    freqs  = np.fft.fftfreq(n, d=1 / sr)[: n // 2]
    return freqs, mag

# Purpose : Computes all 5 features from the FFT output for exploration.
#           No classifier is applied here — the goal is just to see the numbers.
# Input   : signal — 1-D array of amplitude values (from load_audio)
#           sr     — sample rate in Hz
#           freqs  — 1-D array of frequency values in Hz (from compute_fft)
#           mag    — 1-D array of FFT magnitudes (from compute_fft)
# Output  : dictionary with 5 float values, one per feature
def extract_features(signal, sr, freqs, mag):
    # 1. Spectral flatness — how noise-like vs tonal the spectrum is
    mag_safe      = mag + 1e-10
    geo_mean      = np.exp(np.mean(np.log(mag_safe)))
    arith_mean    = np.mean(mag_safe)
    spectral_flat = geo_mean / arith_mean

    # 2. Spectral centroid — centre of mass of the spectrum [Hz]
    spec_centroid = np.sum(freqs * mag) / (np.sum(mag) + 1e-10)

    # 3. High-frequency energy ratio — fraction of energy above 4 kHz
    hf_ratio      = np.sum(mag[freqs > 4000]) / (np.sum(mag) + 1e-10)

    # 4. Spectral rolloff — frequency below which 85% of energy sits [Hz]
    cumsum        = np.cumsum(mag)
    rolloff_idx   = min(np.searchsorted(cumsum, ROLLOFF_PERC * cumsum[-1]), len(freqs) - 1)
    spec_rolloff  = freqs[rolloff_idx]

    # 5. Zero-crossing rate — how often the signal crosses zero per second
    zcr           = np.mean(librosa.feature.zero_crossing_rate(signal)[0])

    return {
        "spectral_flatness": spectral_flat,
        "spectral_centroid": spec_centroid,
        "high_freq_ratio":   hf_ratio,
        "spectral_rolloff":  spec_rolloff,
        "zcr":               zcr,
    }

# Purpose : Runs the full pipeline (load → FFT → features) on every file in a
#           list. No classification is performed — just feature extraction.
#           Stores signal and FFT data so they can be plotted later.
# Input   : file_list — list of filename stems
#           label     — "AI" or "Human", stored for reference in the plots
# Output  : list of result dictionaries, one per file, each containing:
#             name, label, signal, sr, freqs, mag, feats
def process_files(file_list, label):
    results = []
    for fp in file_list:
        name = Path(fp).stem
        try:
            signal, sr = load_audio(resolve_path(fp), DURATION_SEC)
            freqs, mag = compute_fft(signal, sr, FFT_SIZE)
            feats      = extract_features(signal, sr, freqs, mag)
            results.append({
                "name":   name,
                "label":  label,
                "signal": signal,
                "sr":     sr,
                "freqs":  freqs,
                "mag":    mag,
                "feats":  feats,
            })
            print(f"  [OK] {name}")
        except Exception as e:
            print(f"  [ERR] {fp}: {e}")
    return results

# Purpose : Creates a figure showing every sample's time-domain waveform and
#           FFT spectrum side by side, one row per sample. Called twice —
#           once for AI voices, once for Human voices.
# Input   : results     — list of result dictionaries from process_files
#           group_label — title shown at the top (e.g. "AI Voices")
#           color_time  — colour for the waveform plots (hex string)
#           color_fft   — colour for the FFT plots (hex string)
# Output  : none — adds a figure to the matplotlib plot queue
def plot_waveforms_and_ffts(results, group_label, color_time, color_fft):
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(14, 2.5 * n))
    fig.suptitle(f"{group_label} — Time Signals & FFT Spectra", fontsize=14)

    for i, r in enumerate(results):
        time = np.arange(len(r["signal"])) / r["sr"]

        # Time domain
        axes[i, 0].plot(time, r["signal"], linewidth=0.4, color=color_time)
        axes[i, 0].set_title(r["name"], fontsize=9)
        axes[i, 0].set_ylabel("Amplitude", fontsize=8)
        axes[i, 0].set_xlabel("Time [s]", fontsize=8)
        axes[i, 0].tick_params(labelsize=7)
        axes[i, 0].grid(alpha=0.4)

        # Frequency domain
        axes[i, 1].plot(r["freqs"], r["mag"], linewidth=0.4, color=color_fft)
        axes[i, 1].set_title(r["name"], fontsize=9)
        axes[i, 1].set_ylabel("Magnitude", fontsize=8)
        axes[i, 1].set_xlabel("Frequency [Hz]", fontsize=8)
        axes[i, 1].set_xlim(0, 8000)
        axes[i, 1].tick_params(labelsize=7)
        axes[i, 1].grid(alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=3.5)

# Purpose : Box plots showing the distribution of all 5 features across AI vs
#           Human samples. Use this to visually identify which features separate
#           the two groups most clearly.
# Input   : ai_results    — list of result dictionaries for AI files
#           human_results — list of result dictionaries for Human files
# Output  : none — adds a figure to the matplotlib plot queue
def plot_feature_distributions(ai_results, human_results):
    feature_names = list(ai_results[0]["feats"].keys())
    n_feats = len(feature_names)

    fig, axes = plt.subplots(1, n_feats, figsize=(4 * n_feats, 5))
    fig.suptitle("Feature Distributions: AI vs Human", fontsize=14)

    for ax, feat in zip(axes, feature_names):
        ai_vals    = [r["feats"][feat] for r in ai_results]
        human_vals = [r["feats"][feat] for r in human_results]

        bp = ax.boxplot(
            [ai_vals, human_vals],
            labels=["AI", "Human"],
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
        )
        bp["boxes"][0].set_facecolor("#4a90d9")
        bp["boxes"][1].set_facecolor("#e87040")

        for j, vals in enumerate([ai_vals, human_vals], start=1):
            ax.scatter(
                np.full(len(vals), j) + np.random.uniform(-0.08, 0.08, len(vals)),
                vals, zorder=3, s=30,
                color=["#4a90d9", "#e87040"][j - 1], edgecolors="white", linewidths=0.5
            )

        ax.set_title(feat.replace("_", " ").title(), fontsize=10)
        ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()

# Purpose : Renders a colour-coded table showing every sample and all 5 feature
#           values. AI rows are shown in blue, Human rows in orange.
#           There is no prediction column — this is purely for exploring the data.
# Input   : ai_results    — list of result dictionaries for AI files
#           human_results — list of result dictionaries for Human files
# Output  : none — adds a figure to the matplotlib plot queue
def plot_feature_table(ai_results, human_results):
    all_results   = ai_results + human_results
    feature_names = list(all_results[0]["feats"].keys())

    col_labels = ["Sample", "Label"] + \
                 [f.replace("_", " ").title() for f in feature_names]

    rows = []
    for r in all_results:
        row = [r["name"], r["label"]]
        row += [f"{r['feats'][f]:.4f}" for f in feature_names]
        rows.append(row)

    fig, ax = plt.subplots(figsize=(max(14, 2 * len(col_labels)), 0.45 * len(rows) + 2))
    ax.axis("off")
    fig.suptitle("Feature Values — All Samples (no classifier)", fontsize=13, y=0.98)

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Colour header row
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#8b0000")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Colour rows by label: blue for AI, orange for Human
    for i, r in enumerate(all_results, start=1):
        row_color = "#dce8f5" if r["label"] == "AI" else "#fdebd0"
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(row_color)

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
        plot_waveforms_and_ffts(ai_results,    "AI Voices",    "#4a90d9", "#f0c040")
        plot_waveforms_and_ffts(human_results, "Human Voices", "#e87040", "#6abf69")

        # Feature distribution box plots
        plot_feature_distributions(ai_results, human_results)

        # Feature value table
        plot_feature_table(ai_results, human_results)

        plt.show()