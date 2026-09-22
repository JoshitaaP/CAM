import os
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = r"D:\capstone\datasets\SEED_VIG\preprocessed\seed_vig_preprocessed.npz"

print("=" * 70)
print("SEED-VIG PREPROCESSED DATA VERIFICATION")
print("=" * 70)

# ============================================================
# FILE CHECK
# ============================================================

print("\n" + "=" * 70)
print("FILE CHECK")
print("=" * 70)

print("\nExpected file:")
print(os.path.basename(DATA_PATH))

print("\nFile exists:", os.path.exists(DATA_PATH))

if not os.path.exists(DATA_PATH):
    print("\nERROR: Preprocessed file not found.")
    print("Check the path in DATA_PATH.")
    exit()

print("File size: {:.2f} MB".format(os.path.getsize(DATA_PATH) / (1024 * 1024)))

# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING PREPROCESSED DATA")
print("=" * 70)

data = np.load(DATA_PATH)

print("\nVariables stored in file:")

for key in data.files:
    print(f"{key} -> {data[key].shape}")

# ============================================================
# LOAD REQUIRED VARIABLES
# ============================================================

windows = data["windows"]
labels = data["labels"]
subjects = data["subjects"]

# ratings may or may not be present
ratings = data["ratings"] if "ratings" in data.files else None

# ============================================================
# DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("\nWindows shape:")
print(windows.shape)

print("\nLabels shape:")
print(labels.shape)

print("\nSubjects shape:")
print(subjects.shape)

if ratings is not None:
    print("\nRatings shape:")
    print(ratings.shape)

# ============================================================
# SHAPE VERIFICATION
# ============================================================

print("\n" + "=" * 70)
print("SHAPE VERIFICATION")
print("=" * 70)

num_windows = windows.shape[0]
num_channels = windows.shape[1]
samples_per_window = windows.shape[2]

print("\nNumber of windows:", num_windows)
print("Number of channels:", num_channels)
print("Samples per window:", samples_per_window)

print("\nExpected:")
print("Channels: 17")
print("Samples per window: 384")

print("\nWindows shape valid:",
      windows.ndim == 3)

print("Channel count valid:",
      num_channels == 17)

print("Window sample count valid:",
      samples_per_window == 384)

print("Labels length valid:",
      len(labels) == num_windows)

print("Subjects length valid:",
      len(subjects) == num_windows)

shape_passed = (
    windows.ndim == 3
    and num_channels == 17
    and samples_per_window == 384
    and len(labels) == num_windows
    and len(subjects) == num_windows
)

print("\nShape verification:",
      "PASSED ✓" if shape_passed else "FAILED ✗")

# ============================================================
# DATA QUALITY
# ============================================================

print("\n" + "=" * 70)
print("DATA QUALITY CHECK")
print("=" * 70)

print("\nWindows:")
print("NaN values:", np.isnan(windows).sum())
print("Inf values:", np.isinf(windows).sum())

print("\nLabels:")
print("NaN values:", np.isnan(labels).sum())
print("Inf values:", np.isinf(labels).sum())

print("\nSubjects:")
print("NaN values:", np.isnan(subjects).sum())
print("Inf values:", np.isinf(subjects).sum())

nan_count = np.isnan(windows).sum()
inf_count = np.isinf(windows).sum()

quality_passed = nan_count == 0 and inf_count == 0

print("\nData quality:",
      "PASSED ✓" if quality_passed else "FAILED ✗")

# ============================================================
# LABEL VERIFICATION
# ============================================================

print("\n" + "=" * 70)
print("VIGILANCE LABEL VERIFICATION")
print("=" * 70)

unique_labels, label_counts = np.unique(labels, return_counts=True)

print("\nUnique labels:")
print(unique_labels)

print("\nLabel distribution:")

for label, count in zip(unique_labels, label_counts):
    percentage = (count / len(labels)) * 100
    print(
        f"Class {label}: "
        f"{count} windows "
        f"({percentage:.2f}%)"
    )

print("\nExpected classes:")
print("[0 1]")

labels_valid = np.array_equal(
    np.sort(unique_labels),
    np.array([0, 1])
)

print("\nLabel verification:",
      "PASSED ✓" if labels_valid else "FAILED ✗")

# ============================================================
# SUBJECT VERIFICATION
# ============================================================

print("\n" + "=" * 70)
print("SUBJECT VERIFICATION")
print("=" * 70)

unique_subjects, subject_counts = np.unique(
    subjects,
    return_counts=True
)

print("\nNumber of unique subjects:")
print(len(unique_subjects))

print("\nSubject IDs:")
print(unique_subjects)

print("\nWindows per subject:")

for subject, count in zip(unique_subjects, subject_counts):
    print(f"Subject {int(subject):02d}: {count} windows")

# ============================================================
# NORMALIZATION CHECK
# ============================================================

print("\n" + "=" * 70)
print("NORMALIZATION CHECK")
print("=" * 70)

global_mean = np.mean(windows)
global_std = np.std(windows)

print("\nGlobal mean:")
print(global_mean)

print("\nGlobal standard deviation:")
print(global_std)

print("\nExpected:")
print("Mean approximately 0")
print("Std approximately 1")

mean_valid = np.isclose(global_mean, 0, atol=0.01)
std_valid = np.isclose(global_std, 1, atol=0.01)

print("\nMean normalization valid:", mean_valid)
print("Standard deviation valid:", std_valid)

print(
    "\nNormalization:",
    "PASSED ✓" if mean_valid and std_valid else "CHECK ✗"
)

# ============================================================
# VALUE RANGE
# ============================================================

print("\n" + "=" * 70)
print("VALUE RANGE CHECK")
print("=" * 70)

print("\nMinimum EEG value:")
print(np.min(windows))

print("\nMaximum EEG value:")
print(np.max(windows))

print("\nMean absolute value:")
print(np.mean(np.abs(windows)))

print("\n95th percentile absolute value:")
print(np.percentile(np.abs(windows), 95))

print("\n99th percentile absolute value:")
print(np.percentile(np.abs(windows), 99))

print("\n99.9th percentile absolute value:")
print(np.percentile(np.abs(windows), 99.9))

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL VERIFICATION SUMMARY")
print("=" * 70)

print("\nChannels:", num_channels)
print("Samples per window:", samples_per_window)
print("Total windows:", num_windows)
print("Unique subjects:", len(unique_subjects))
print("Unique labels:", unique_labels)

print("\nNaN values:", np.isnan(windows).sum())
print("Inf values:", np.isinf(windows).sum())

print("\n" + "=" * 70)

if shape_passed and quality_passed and labels_valid:
    print("SEED-VIG BASIC VERIFICATION PASSED ✓")
else:
    print("SEED-VIG VERIFICATION REQUIRES ATTENTION ✗")

print("=" * 70)