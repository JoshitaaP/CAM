import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt
import os


# ============================================================
# CONFIGURATION
# ============================================================

FILE_PATH = r"D:\capstone\datasets\SEED_VIG.mat"

OUTPUT_DIR = r"D:\capstone\datasets\SEED_VIG\preprocessed"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "seed_vig_preprocessed.npz"
)

SAMPLING_RATE = 128

LOW_CUTOFF = 1.0
HIGH_CUTOFF = 45.0

FILTER_ORDER = 4


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("SEED-VIG EEG PREPROCESSING")
print("=" * 70)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading SEED-VIG dataset...")

data = sio.loadmat(FILE_PATH)

eeg = data["EEGsample"]
subject_ids = data["subindex"].flatten()
states = data["substate"].flatten()

print("Original EEG shape:", eeg.shape)
print("Subject ID shape:", subject_ids.shape)
print("State shape:", states.shape)


# ============================================================
# DATASET INFORMATION
# ============================================================

num_samples = eeg.shape[0]
num_channels = eeg.shape[1]
num_timepoints = eeg.shape[2]

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("Samples        :", num_samples)
print("Channels       :", num_channels)
print("Timepoints     :", num_timepoints)
print("Sampling rate  :", SAMPLING_RATE, "Hz")

duration = num_timepoints / SAMPLING_RATE

print("Sample duration:", duration, "seconds")


# ============================================================
# DATA QUALITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("DATA QUALITY CHECK")
print("=" * 70)

nan_count = np.isnan(eeg).sum()
inf_count = np.isinf(eeg).sum()

print("NaN values :", nan_count)
print("Inf values :", inf_count)

if nan_count == 0 and inf_count == 0:
    print("Data quality check: PASSED ✓")
else:
    print("WARNING: Invalid values detected!")


# ============================================================
# ORIGINAL DATA STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("ORIGINAL EEG STATISTICS")
print("=" * 70)

print("Minimum :", np.min(eeg))
print("Maximum :", np.max(eeg))
print("Mean    :", np.mean(eeg))
print("Std     :", np.std(eeg))


# ============================================================
# SUBJECT INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("SUBJECT INFORMATION")
print("=" * 70)

unique_subjects = np.unique(subject_ids)

print("Number of subjects:", len(unique_subjects))

for subject in unique_subjects:

    count = np.sum(subject_ids == subject)

    print(
        f"Subject {int(subject):02d}: "
        f"{count} samples"
    )


# ============================================================
# STATE INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("STATE INFORMATION")
print("=" * 70)

unique_states, state_counts = np.unique(
    states,
    return_counts=True
)

for state, count in zip(unique_states, state_counts):

    percentage = (count / len(states)) * 100

    print(
        f"State {int(state)}: "
        f"{count} samples "
        f"({percentage:.2f}%)"
    )


# ============================================================
# FILTER DESIGN
# ============================================================

print("\n" + "=" * 70)
print("FILTER DESIGN")
print("=" * 70)

print("Filter type :", "Butterworth band-pass")
print("Order       :", FILTER_ORDER)
print("Low cutoff  :", LOW_CUTOFF, "Hz")
print("High cutoff :", HIGH_CUTOFF, "Hz")


nyquist = SAMPLING_RATE / 2

low = LOW_CUTOFF / nyquist
high = HIGH_CUTOFF / nyquist

b, a = butter(
    FILTER_ORDER,
    [low, high],
    btype="band"
)


# ============================================================
# PREPROCESSING FUNCTION
# ============================================================

def preprocess_sample(sample):

    """
    Preprocess one EEG sample.

    Input:
        sample -> (channels, timepoints)

    Output:
        processed -> (channels, timepoints)
    """

    processed = np.zeros_like(
        sample,
        dtype=np.float64
    )

    for channel in range(sample.shape[0]):

        signal = sample[channel]

        # --------------------------------------------
        # Band-pass filtering
        # --------------------------------------------

        filtered = filtfilt(
            b,
            a,
            signal
        )

        # --------------------------------------------
        # Z-score normalization
        # --------------------------------------------

        mean = np.mean(filtered)
        std = np.std(filtered)

        if std > 1e-8:

            normalized = (
                filtered - mean
            ) / std

        else:

            normalized = (
                filtered - mean
            )

        processed[channel] = normalized

    return processed


# ============================================================
# PREPROCESSING TEST - SAMPLE 1
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING TEST - SAMPLE 1")
print("=" * 70)

sample_1 = eeg[0]

processed_sample_1 = preprocess_sample(
    sample_1
)

print("Original shape  :", sample_1.shape)
print("Processed shape :", processed_sample_1.shape)


print("\nChannel 1 after preprocessing:")

print(
    "Mean:",
    np.mean(processed_sample_1[0])
)

print(
    "Std :",
    np.std(processed_sample_1[0])
)

print(
    "Min :",
    np.min(processed_sample_1[0])
)

print(
    "Max :",
    np.max(processed_sample_1[0])
)


# ============================================================
# VISUALIZE ORIGINAL VS PREPROCESSED
# ============================================================

time = np.arange(num_timepoints) / SAMPLING_RATE

plt.figure(figsize=(12, 5))

plt.plot(
    time,
    sample_1[0],
    label="Original"
)

plt.title(
    "SEED-VIG S01 - Channel 1 - Original"
)

plt.xlabel("Time (seconds)")
plt.ylabel("EEG Amplitude")

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 5))

plt.plot(
    time,
    processed_sample_1[0],
    label="Preprocessed"
)

plt.title(
    "SEED-VIG S01 - Channel 1 - Preprocessed"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Normalized Amplitude")

plt.tight_layout()
plt.show()


# ============================================================
# PROCESS ALL SAMPLES
# ============================================================

print("\n" + "=" * 70)
print("PROCESSING ALL SEED-VIG SAMPLES")
print("=" * 70)

processed_eeg = np.zeros_like(
    eeg,
    dtype=np.float64
)


for i in range(num_samples):

    processed_eeg[i] = preprocess_sample(
        eeg[i]
    )

    print(
        f"Sample {i + 1:04d}/{num_samples} processed"
    )


# ============================================================
# FINAL SHAPE
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSED DATA SHAPE")
print("=" * 70)

print(
    "Processed EEG shape:",
    processed_eeg.shape
)

print(
    "Expected:",
    (num_samples, num_channels, num_timepoints)
)


# ============================================================
# POST-PROCESSING STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("POST-PREPROCESSING STATISTICS")
print("=" * 70)

print(
    "Global minimum:",
    np.min(processed_eeg)
)

print(
    "Global maximum:",
    np.max(processed_eeg)
)

print(
    "Global mean:",
    np.mean(processed_eeg)
)

print(
    "Global standard deviation:",
    np.std(processed_eeg)
)


# ============================================================
# DATA QUALITY AFTER PREPROCESSING
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATA QUALITY CHECK")
print("=" * 70)

nan_count = np.isnan(processed_eeg).sum()
inf_count = np.isinf(processed_eeg).sum()

print("NaN values:", nan_count)
print("Inf values:", inf_count)

if nan_count == 0 and inf_count == 0:

    print("Data quality: PASSED ✓")

else:

    print("WARNING: Invalid values detected!")


# ============================================================
# SUBJECT IDS / LABELS
# ============================================================

print("\n" + "=" * 70)
print("LABEL AND SUBJECT VERIFICATION")
print("=" * 70)

print(
    "EEG samples:",
    processed_eeg.shape[0]
)

print(
    "Subject IDs:",
    subject_ids.shape
)

print(
    "States:",
    states.shape
)

print(
    "Unique subjects:",
    len(np.unique(subject_ids))
)

print(
    "Unique states:",
    np.unique(states)
)


# ============================================================
# SAVE DATA
# ============================================================

print("\n" + "=" * 70)
print("SAVING PREPROCESSED DATA")
print("=" * 70)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

np.savez_compressed(
    OUTPUT_FILE,
    eeg=processed_eeg,
    labels=states,
    subjects=subject_ids,
    sampling_rate=SAMPLING_RATE,
    sample_duration=duration,
    channels=num_channels,
    timepoints=num_timepoints
)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nOutput file:")
print(OUTPUT_FILE)

print(
    "\nFile exists:",
    os.path.exists(OUTPUT_FILE)
)

if os.path.exists(OUTPUT_FILE):

    size_mb = (
        os.path.getsize(OUTPUT_FILE)
        / (1024 * 1024)
    )

    print(
        f"File size: {size_mb:.2f} MB"
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SEED-VIG PREPROCESSING COMPLETED SUCCESSFULLY ✓")
print("=" * 70)

print("Subjects       :", len(unique_subjects))
print("EEG channels   :", num_channels)
print("Sampling rate  :", SAMPLING_RATE, "Hz")
print("Sample duration:", duration, "seconds")
print("Samples        :", num_samples)
print("Timepoints     :", num_timepoints)
print("States         :", len(unique_states))
print("Output         :", OUTPUT_FILE)

print("=" * 70)