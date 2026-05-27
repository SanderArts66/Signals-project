import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Replace audio.wav with your own audio file
sample_rate, audio = wavfile.read("audio.wav")

# Convert stereo to mono if needed
if len(audio.shape) > 1:
    audio = audio.mean(axis=1)

# Normalize audio
audio = audio / np.max(np.abs(audio))

# Take first 2 seconds
duration = 2  # seconds
N = duration * sample_rate
signal = audio[:N]

# Compute DFT
dft = np.fft.fft(signal)

# Frequency axis
frequencies = np.fft.fftfreq(len(signal), d=1/sample_rate)

# Magnitude spectrum
magnitude = np.abs(dft)

# Keep only positive frequencies
half = len(frequencies) // 2
frequencies = frequencies[:half]
magnitude = magnitude[:half]

# Plot time domain signal
time = np.arange(len(signal)) / sample_rate

plt.figure(figsize=(12, 5))
plt.plot(time, signal)
plt.title("Time Domain Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

# Plot frequency domain
plt.figure(figsize=(12, 5))
plt.plot(frequencies, magnitude)
plt.title("DFT Magnitude Spectrum")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.xlim(0, 8000)  # Speech frequencies
plt.grid()
plt.show()

# Feature 1: High Frequency Energy Ratio
high_freq_energy = np.sum(magnitude[frequencies > 4000])
total_energy = np.sum(magnitude)

high_freq_ratio = high_freq_energy / total_energy

# Feature 2: Spectral Flatness
# Avoid log(0)
magnitude_safe = magnitude + 1e-10

geometric_mean = np.exp(np.mean(np.log(magnitude_safe)))
arithmetic_mean = np.mean(magnitude_safe)

spectral_flatness = geometric_mean / arithmetic_mean

# Print features
print("\n========== FEATURE VALUES ==========")
print(f"High-frequency energy ratio: {high_freq_ratio:.4f}")
print(f"Spectral flatness: {spectral_flatness:.4f}")

# These thresholds are experimental
# We can tune them based on results
if high_freq_ratio < 0.15 and spectral_flatness > 0.2:
    prediction = "AI-generated voice"
else:
    prediction = "Human voice"

# Output results
print("\n========== DETECTION RESULT ==========")
print(f"Prediction: {prediction}")
