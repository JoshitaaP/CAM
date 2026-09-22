import pickle
file_path=r"D:\capstone\datasets\deap-1\deap-dataset\data_preprocessed_python\s01.dat"
with open(file_path,"rb") as f:
    subject =pickle.load(f,encoding="latin1")
print("Successfully loaded DEAP subject S01!")
print("Keys:",subject.keys())

print("\nData shape:", subject["data"].shape)
print("Labels shape:", subject["labels"].shape)

print("\nFirst trial data shape:", subject["data"][0].shape)
print("First trial labels:", subject["labels"][0])

import matplotlib.pyplot as plt
import numpy as np

eeg_signal = subject["data"][0, 0, :]

print("Signal shape:", eeg_signal.shape)
print("First 10 values:", eeg_signal[:10])
print("Minimum:", np.min(eeg_signal))
print("Maximum:", np.max(eeg_signal))
print("Mean:", np.mean(eeg_signal))
print("NaN values:", np.isnan(eeg_signal).sum())

plt.figure(figsize=(12, 4))
plt.plot(eeg_signal[:500])
plt.title("DEAP - S01, Trial 1, EEG Channel 1")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

sampling_rate = 128

num_samples = subject["data"].shape[2]
duration = num_samples / sampling_rate

print("\nSampling rate:", sampling_rate, "Hz")
print("Number of samples:", num_samples)
print("Trial duration:", duration, "seconds")
print("Trial duration:", duration / 60, "minutes")

# Inspect the first sample of all 40 channels
print("\nNumber of channels:", subject["data"].shape[1])

for i in range(40):
    print(f"Channel {i}: mean = {subject['data'][0, i, :].mean():.4f}")
# Visualize the first 4 EEG channels from Trial 1
plt.figure(figsize=(12, 8))

for i in range(4):
    plt.plot(
        subject["data"][0, i, :500],
        label=f"Channel {i}"
    )

plt.title("DEAP S01 - Trial 1 - First 4 EEG Channels")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()
eeg_data = subject["data"][0, :32, :]

print("\nEEG data shape:", eeg_data.shape)

for i in range(32):
    print(
        f"Channel {i}: "
        f"mean={eeg_data[i].mean():.4f}, "
        f"std={eeg_data[i].std():.4f}, "
        f"min={eeg_data[i].min():.4f}, "
        f"max={eeg_data[i].max():.4f}"
    )
# ==================================================
# DEAP LABEL ANALYSIS
# ==================================================

labels = subject["labels"]

valence = labels[:, 0]
arousal = labels[:, 1]
dominance = labels[:, 2]
liking = labels[:, 3]

print("\n" + "=" * 50)
print("DEAP LABEL ANALYSIS")
print("=" * 50)

print("\nNumber of trials:", len(labels))

print("\nValence:")
print("Minimum:", np.min(valence))
print("Maximum:", np.max(valence))
print("Mean:", np.mean(valence))
print("Std:", np.std(valence))

print("\nArousal:")
print("Minimum:", np.min(arousal))
print("Maximum:", np.max(arousal))
print("Mean:", np.mean(arousal))
print("Std:", np.std(arousal))

print("\nDominance:")
print("Minimum:", np.min(dominance))
print("Maximum:", np.max(dominance))
print("Mean:", np.mean(dominance))
print("Std:", np.std(dominance))

print("\nLiking:")
print("Minimum:", np.min(liking))
print("Maximum:", np.max(liking))
print("Mean:", np.mean(liking))
print("Std:", np.std(liking))

# ============================================================
# DEAP LABEL DISTRIBUTION EDA
# ============================================================

labels = subject["labels"]

valence = labels[:, 0]
arousal = labels[:, 1]
dominance = labels[:, 2]
liking = labels[:, 3]

print("\n" + "=" * 50)
print("DEAP LABEL DISTRIBUTION")
print("=" * 50)

# ------------------------------------------------------------
# 1. Valence distribution
# ------------------------------------------------------------

print("\nVALENCE")

print("Low valence (< 5):", np.sum(valence < 5))
print("High valence (>= 5):", np.sum(valence >= 5))

# ------------------------------------------------------------
# 2. Arousal distribution
# ------------------------------------------------------------

print("\nAROUSAL")

print("Low arousal (< 5):", np.sum(arousal < 5))
print("High arousal (>= 5):", np.sum(arousal >= 5))

# ------------------------------------------------------------
# 3. Dominance distribution
# ------------------------------------------------------------

print("\nDOMINANCE")

print("Low dominance (< 5):", np.sum(dominance < 5))
print("High dominance (>= 5):", np.sum(dominance >= 5))

# ------------------------------------------------------------
# 4. Liking distribution
# ------------------------------------------------------------

print("\nLIKING")

print("Low liking (< 5):", np.sum(liking < 5))
print("High liking (>= 5):", np.sum(liking >= 5))


# ============================================================
# HISTOGRAMS
# ============================================================

plt.figure(figsize=(10, 5))
plt.hist(valence, bins=10)
plt.title("DEAP S01 - Valence Distribution")
plt.xlabel("Valence")
plt.ylabel("Number of Trials")
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 5))
plt.hist(arousal, bins=10)
plt.title("DEAP S01 - Arousal Distribution")
plt.xlabel("Arousal")
plt.ylabel("Number of Trials")
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 5))
plt.hist(dominance, bins=10)
plt.title("DEAP S01 - Dominance Distribution")
plt.xlabel("Dominance")
plt.ylabel("Number of Trials")
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 5))
plt.hist(liking, bins=10)
plt.title("DEAP S01 - Liking Distribution")
plt.xlabel("Liking")
plt.ylabel("Number of Trials")
plt.grid(True)
plt.show()


# ============================================================
# VALENCE vs AROUSAL
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(valence, arousal)

plt.axvline(5, linestyle="--")
plt.axhline(5, linestyle="--")

plt.title("DEAP S01 - Valence vs Arousal")
plt.xlabel("Valence")
plt.ylabel("Arousal")
plt.grid(True)
plt.show()