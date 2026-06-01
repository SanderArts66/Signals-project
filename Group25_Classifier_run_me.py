import numpy as np
import matplotlib.pyplot as plt
import librosa


# ── Variables ────────────────────────────────────────────────────────────────


DURATION_SEC = 5      # seconds of audio to analyse per file, if audio is longer than 10 min,
# a FFT_size should be implemented here and in function compute_fft. However, since the
#application does not need more than 10 min to determine AI or human, this was left out.

ROLLOFF_PERC = 0.85   # spectral rolloff threshold, could be changed, however since we are
# comparing Ai relative to Human, the exact value matters little.

FLATNESS_THRESHOLD = 0.10   


# ── Input the file you want to use ────────────────────────────────────────────


Input_File         = "Ai1.wav"  


# ── Main functions (notice that the first 3 functions are the same as in Research file) ──


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


def compute_fft(signal, sample_rate):
    N         = len(signal)
    window    = np.hanning(N)
    dft       = np.fft.fft(signal * window, n=N)
    magnitude = np.abs(dft[: N // 2])
    freqs     = np.fft.fftfreq(N, d=1 / sample_rate)[: N // 2]
    return freqs, magnitude


def extract_features(freqs, mag):
    magnitude_safe  = mag + 1e-10
    geometric_mean  = np.exp(np.mean(np.log(magnitude_safe)))
    arithmetic_mean = np.mean(magnitude_safe)
    spectral_flat   = geometric_mean / arithmetic_mean

    spectral_centroid = np.sum(freqs * mag) / np.sum(mag)

    hf_ratio = np.sum(mag[freqs > 4000]) / np.sum(mag)

    cumsum      = np.cumsum(mag)
    rolloff_idx = min(np.searchsorted(cumsum, ROLLOFF_PERC * cumsum[-1]), len(freqs) - 1)
    spec_rolloff = freqs[rolloff_idx]

    spec_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * mag) / np.sum(mag))

    return {
        "spectral_flatness":  spectral_flat,
        "spectral_centroid":  spectral_centroid,
        "high_freq_ratio":    hf_ratio,
        "spectral_rolloff":   spec_rolloff,
        "spectral_bandwidth": spec_bandwidth}


def classify(features):
    score = 0

    if features["spectral_flatness"] > 0.1:
        score += 3
    else:
        score -= 0

    if features["spectral_rolloff"] > 6000:
        score += 1
    else:
        score -= 0

    if features["spectral_centroid"] > 2400:
        score += 1
    else:
        score -= 0

    if features["high_freq_ratio"] > 0.24:
        score += 1
    else:
        score -= 0

    if features["spectral_bandwidth"] > 2800:
        score += 1
    else:
        score -= 0

    if score >= 4:
        return "AI"
    else: 
        return "Human"
 

# ── Plots ──────────────────────────────────────────────────────────────────


def plot_signal(signal, sr, freqs, mag, name, prediction):
    color = "#e84040" if prediction == "AI" else \
            "#40a060" if prediction == "Human" else "#e8a040"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{name}  —  Prediction: {prediction}", fontsize=14, color=color)

    time = np.arange(len(signal)) / sr
    ax1.plot(time, signal, linewidth=0.5, color=color)
    ax1.set_title("Time Signal")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Amplitude")
    ax1.grid(alpha=0.4)

    ax2.plot(freqs, mag, linewidth=0.5, color=color)
    ax2.set_title("FFT Spectrum")
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("Magnitude")
    ax2.set_xlim(0, 8000)
    ax2.grid(alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.93])


# ── Prints in terminal ────────────────────────────────────────────────────


def print_results(name, features, prediction):
    print("\n" + "=" * 55)
    print(f"  File       : {name}")
    print(f"  Prediction : {prediction}")
    print("-" * 55)
    print(f"  {'Feature':<24} {'Value':>10}   {'Threshold':>9}")
    print(f"  {'-'*24} {'-'*10}   {'-'*9}")
    print(f"  {'Spectral Flatness':<24} {features['spectral_flatness']:>10.5f}   {FLATNESS_THRESHOLD:>9.5f}")
    print(f"  {'Spectral Centroid':<24} {features['spectral_centroid']:>10.1f}   {'2750.0':>9}")
    print(f"  {'High Freq Ratio':<24} {features['high_freq_ratio']:>10.5f}   {'0.28000':>9}")
    print(f"  {'Spectral Rolloff':<24} {features['spectral_rolloff']:>10.1f}   {'7250.0':>9}")
    print(f"  {'Spectral Bandwidth':<24} {features['spectral_bandwidth']:>10.1f}   {'3000.0':>9}")
    print("=" * 55 + "\n")


# ── Stress test 1: Parameter sensitivity ─────────────────────────────────────


def parameter_sensitivity_test(filepath):
    durations = [1, 2, 3, 5]
    flatness_values = []

    print("\n" + "=" * 45)
    print("PARAMETER SENSITIVITY TEST")
    print("=" * 45)

    for dur in durations:
        signal, sr = load_audio(filepath, dur)
        freqs, mag = compute_fft(signal, sr)
        features   = extract_features(freqs, mag)
        flatness_values.append(features["spectral_flatness"])
        print(f"Duration = {dur}s | Flatness = {features['spectral_flatness']:.5f}")

    plt.figure(figsize=(8, 4))
    plt.plot(durations, flatness_values, marker="o", label="Spectral Flatness")
    plt.xlabel("Observation Window Length (s)")
    plt.ylabel("Spectral flatness")
    plt.title("Parameter Sensitivity Test")
    plt.grid(alpha=0.4)
    plt.legend()


# ── Stress test 2: Noise robustness ──────────────────────────────────────────


def add_noise(signal, snr_db):
    signal_power = np.mean(signal ** 2)
    noise_power  = signal_power / (10 ** (snr_db / 10))
    noise        = np.random.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise

def noise_robustness_test(filepath):
    signal, sr = load_audio(filepath, DURATION_SEC)
    snr_values  = [40, 30, 20, 10, 0]
    flatness_values = []

    print("\n" + "=" * 45)
    print("NOISE ROBUSTNESS TEST")
    print("=" * 45)

    for snr in snr_values:
        noisy_signal = add_noise(signal, snr)
        freqs, mag   = compute_fft(noisy_signal, sr)
        features     = extract_features(freqs, mag)
        prediction   = classify(features)
        flatness_values.append(features["spectral_flatness"])
        print(f"SNR = {snr:>2} dB | Flatness = {features['spectral_flatness']:.5f} | Prediction = {prediction}")

    plt.figure(figsize=(8, 4))
    plt.plot(snr_values, flatness_values, marker="o", label="Spectral Flatness")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Spectral Flatness")
    plt.title("Noise Robustness Test")
    plt.grid(alpha=0.4)
    plt.legend()

# ── Main Text ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    filepath = Input_File
    name = filepath.split(".")[0]

    try:
        signal, sample_rate = load_audio(filepath, DURATION_SEC)
        freqs, magnitude    = compute_fft(signal, sample_rate)
        features            = extract_features(freqs, magnitude)
    except Exception as e:
        print(f"[ERR] Could not load file: {e}")
        exit(1)

    prediction = classify(features)

    plot_signal(signal, sample_rate, freqs, magnitude, name, prediction)
    print_results(name, features, prediction)

    parameter_sensitivity_test(filepath)
    noise_robustness_test(filepath)

    plt.show()