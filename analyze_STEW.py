import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# STEW DATASET - COMPLETE EDA
# ============================================================

BASE_PATH = r"D:\capstone\datasets\STEW"

DATA_FILE = BASE_PATH + r"\dataset.mat"
RATING_FILE = BASE_PATH + r"\rating.mat"
CLASS_FILE = BASE_PATH + r"\class_012.mat"
ONE_HOT_FILE = BASE_PATH + r"\three_class_one_hot.mat"


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 70)
print("STEW DATASET - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

data = sio.loadmat(DATA_FILE)

dataset = data["dataset"]

print("\nDataset keys:")
print(data.keys())

print("\nDataset shape:")
print(dataset.shape)

print("Expected format:")
print("(Channels, Samples, Subjects)")


# ============================================================
# 2. BASIC DATASET INFORMATION
# ============================================================

num_channels = dataset.shape[0]
num_samples = dataset.shape[1]
num_subjects = dataset.shape[2]

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("Number of subjects :", num_subjects)
print("Number of channels :", num_channels)
print("Number of samples  :", num_samples)


# ============================================================
# 3. CHECK DATA TYPE
# ============================================================

print("\nData type:")
print(dataset.dtype)


# ============================================================
# 4. CHECK NaN AND INF
# ============================================================

print("\n" + "=" * 70)
print("DATA QUALITY CHECK")
print("=" * 70)

nan_count = np.isnan(dataset).sum()
inf_count = np.isinf(dataset).sum()

print("Total NaN values :", nan_count)
print("Total Inf values :", inf_count)

if nan_count == 0 and inf_count == 0:
    print("Data quality check: PASSED ✓")
else:
    print("WARNING: NaN or Inf values detected!")


# ============================================================
# 5. GLOBAL STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("GLOBAL EEG STATISTICS")
print("=" * 70)

print("Minimum :", np.min(dataset))
print("Maximum :", np.max(dataset))
print("Mean    :", np.mean(dataset))
print("Std     :", np.std(dataset))


# ============================================================
# 6. SUBJECT 1 STATISTICS
# ============================================================

subject_1 = dataset[:, :, 0]

print("\n" + "=" * 70)
print("SUBJECT 1 STATISTICS")
print("=" * 70)

print("Subject 1 shape:", subject_1.shape)

print("Minimum :", np.min(subject_1))
print("Maximum :", np.max(subject_1))
print("Mean    :", np.mean(subject_1))
print("Std     :", np.std(subject_1))

print("NaN values:", np.isnan(subject_1).sum())
print("Inf values:", np.isinf(subject_1).sum())


# ============================================================
# 7. CHANNEL-WISE STATISTICS - SUBJECT 1
# ============================================================

print("\n" + "=" * 70)
print("CHANNEL-WISE STATISTICS - SUBJECT 1")
print("=" * 70)

for channel in range(num_channels):

    signal = subject_1[channel]

    print(
        f"Channel {channel + 1:02d}: "
        f"mean={np.mean(signal):.4f}, "
        f"std={np.std(signal):.4f}, "
        f"min={np.min(signal):.4f}, "
        f"max={np.max(signal):.4f}"
    )


# ============================================================
# 8. SUBJECT-WISE STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("SUBJECT-WISE STATISTICS")
print("=" * 70)

for subject in range(num_subjects):

    subject_data = dataset[:, :, subject]

    print(
        f"Subject {subject + 1:02d}: "
        f"mean={np.mean(subject_data):.4f}, "
        f"std={np.std(subject_data):.4f}, "
        f"min={np.min(subject_data):.4f}, "
        f"max={np.max(subject_data):.4f}"
    )


# ============================================================
# 9. LOAD LABELS
# ============================================================

rating_data = sio.loadmat(RATING_FILE)
class_data = sio.loadmat(CLASS_FILE)
one_hot_data = sio.loadmat(ONE_HOT_FILE)

ratings = rating_data["rating"].flatten()
classes = class_data["class_012"].flatten()
one_hot = one_hot_data["three_class_one_hot"]


# ============================================================
# 10. LABEL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("LABEL INFORMATION")
print("=" * 70)

print("Ratings shape :", ratings.shape)
print("Classes shape :", classes.shape)
print("One-hot shape :", one_hot.shape)

print("\nRatings:")
print(ratings)

print("\nClasses:")
print(classes)

print("\nOne-hot labels:")
print(one_hot)


# ============================================================
# 11. CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("STEW CLASS DISTRIBUTION")
print("=" * 70)

unique_classes, class_counts = np.unique(
    classes,
    return_counts=True
)

for cls, count in zip(unique_classes, class_counts):

    percentage = (count / len(classes)) * 100

    print(
        f"Class {cls}: "
        f"{count} subjects "
        f"({percentage:.2f}%)"
    )


# ============================================================
# 12. RATING STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("RATING STATISTICS")
print("=" * 70)

print("Minimum :", np.min(ratings))
print("Maximum :", np.max(ratings))
print("Mean    :", np.mean(ratings))
print("Std     :", np.std(ratings))


# ============================================================
# 13. RATING DISTRIBUTION
# ============================================================

unique_ratings, rating_counts = np.unique(
    ratings,
    return_counts=True
)

print("\nRating frequency:")

for rating, count in zip(unique_ratings, rating_counts):

    print(
        f"Rating {rating}: "
        f"{count} subjects"
    )


# ============================================================
# 14. SUBJECT 1 - CHANNEL 1 SIGNAL
# ============================================================

channel_1 = dataset[0, :, 0]

print("\n" + "=" * 70)
print("SUBJECT 1 - CHANNEL 1")
print("=" * 70)

print("Signal shape :", channel_1.shape)
print("Minimum      :", np.min(channel_1))
print("Maximum      :", np.max(channel_1))
print("Mean         :", np.mean(channel_1))
print("Std          :", np.std(channel_1))


# ============================================================
# 15. PLOT COMPLETE SIGNAL
# ============================================================

plt.figure(figsize=(14, 5))

plt.plot(channel_1)

plt.title(
    "STEW - Subject 1 - Channel 1 - Complete EEG Signal"
)

plt.xlabel("Sample")
plt.ylabel("Amplitude")

plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 16. PLOT FIRST 1000 SAMPLES
# ============================================================

plt.figure(figsize=(14, 5))

plt.plot(channel_1[:1000])

plt.title(
    "STEW - Subject 1 - Channel 1 - First 1000 Samples"
)

plt.xlabel("Sample")
plt.ylabel("Amplitude")

plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 17. PLOT FIRST 4 CHANNELS
# ============================================================

plt.figure(figsize=(14, 8))

for channel in range(min(4, num_channels)):

    plt.plot(
        subject_1[channel, :1000],
        label=f"Channel {channel + 1}"
    )

plt.title(
    "STEW - Subject 1 - First 4 EEG Channels"
)

plt.xlabel("Sample")
plt.ylabel("Amplitude")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 18. CHANNEL STANDARD DEVIATION
# ============================================================

channel_stds = []

for channel in range(num_channels):

    channel_stds.append(
        np.std(subject_1[channel])
    )

plt.figure(figsize=(12, 5))

plt.bar(
    range(1, num_channels + 1),
    channel_stds
)

plt.title(
    "STEW - Subject 1 - Channel Standard Deviations"
)

plt.xlabel("Channel")
plt.ylabel("Standard Deviation")

plt.grid(axis="y")
plt.tight_layout()
plt.show()


# ============================================================
# 19. SUBJECT STANDARD DEVIATION
# ============================================================

subject_stds = []

for subject in range(num_subjects):

    subject_stds.append(
        np.std(dataset[:, :, subject])
    )

plt.figure(figsize=(14, 5))

plt.bar(
    range(1, num_subjects + 1),
    subject_stds
)

plt.title(
    "STEW - Standard Deviation Across Subjects"
)

plt.xlabel("Subject")
plt.ylabel("Standard Deviation")

plt.grid(axis="y")
plt.tight_layout()
plt.show()


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEW EDA SUMMARY")
print("=" * 70)

print("Subjects          :", num_subjects)
print("EEG channels      :", num_channels)
print("Samples/subject   :", num_samples)

print("\nData quality:")
print("NaN values        :", nan_count)
print("Inf values        :", inf_count)

print("\nClasses:")

for cls, count in zip(unique_classes, class_counts):

    print(
        f"Class {cls}: "
        f"{count} subjects"
    )

print("\nRatings:")
print("Minimum           :", np.min(ratings))
print("Maximum           :", np.max(ratings))
print("Mean              :", np.mean(ratings))
print("Std               :", np.std(ratings))

print("\nSTEW EDA completed successfully! ✓")