import os
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt


# ============================================================
# STEW EEG PREPROCESSING
# ============================================================

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_PATH = r"D:\capstone\datasets\STEW"

DATA_FILE = os.path.join(BASE_PATH, "dataset.mat")
CLASS_FILE = os.path.join(BASE_PATH, "class_012.mat")
RATING_FILE = os.path.join(BASE_PATH, "rating.mat")

OUTPUT_DIR = os.path.join(BASE_PATH, "preprocessed")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------

FS = 128

WINDOW_DURATION = 2

SAMPLES_PER_WINDOW = FS * WINDOW_DURATION

LOW_CUT = 1.0
HIGH_CUT = 45.0

FILTER_ORDER = 4

EXPECTED_CHANNELS = 14
EXPECTED_SUBJECTS = 45


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 70)
print("STEW EEG PREPROCESSING")
print("=" * 70)

print("\nLoading STEW dataset...")

data = sio.loadmat(DATA_FILE)

dataset = data["dataset"]

print("Original dataset shape:", dataset.shape)

print("\nExpected format:")
print("(14 channels, 19200 samples, 45 subjects)")


# ============================================================
# 2. LOAD LABELS
# ============================================================

class_data = sio.loadmat(CLASS_FILE)

classes = class_data["class_012"].flatten()

print("\nClass labels shape:", classes.shape)

print("Class values:", np.unique(classes))


# ------------------------------------------------------------
# Load ratings
# ------------------------------------------------------------

ratings = None

if os.path.exists(RATING_FILE):

    rating_data = sio.loadmat(RATING_FILE)

    print("\nRating file found.")

    print("Rating file variables:")

    for key, value in rating_data.items():

        if not key.startswith("__"):

            print(
                key,
                "| shape:",
                value.shape
            )

    # Try to find the actual rating variable
    for key, value in rating_data.items():

        if not key.startswith("__"):

            if isinstance(value, np.ndarray):

                if value.size >= EXPECTED_SUBJECTS:

                    ratings = value.flatten()

                    break

else:

    print("\nRating file not found.")
    print("Continuing without ratings.")


# ============================================================
# 3. DATASET INFORMATION
# ============================================================

num_channels = dataset.shape[0]
num_samples = dataset.shape[1]
num_subjects = dataset.shape[2]

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("Subjects :", num_subjects)
print("Channels :", num_channels)
print("Samples  :", num_samples)
print("Sampling rate :", FS, "Hz")

if num_channels != EXPECTED_CHANNELS:

    raise ValueError(
        f"Expected {EXPECTED_CHANNELS} channels "
        f"but found {num_channels}"
    )


# ============================================================
# 4. FILTER DESIGN
# ============================================================

print("\n" + "=" * 70)
print("FILTER DESIGN")
print("=" * 70)

nyquist = FS / 2

low = LOW_CUT / nyquist
high = HIGH_CUT / nyquist

b, a = butter(
    FILTER_ORDER,
    [low, high],
    btype="band"
)

print("Filter type : Butterworth band-pass")
print("Order       :", FILTER_ORDER)
print("Low cutoff  :", LOW_CUT, "Hz")
print("High cutoff :", HIGH_CUT, "Hz")


# ============================================================
# 5. PREPROCESSING FUNCTION
# ============================================================

def preprocess_subject(subject_data):

    """
    Input:
        subject_data
        Shape = (14, 19200)

    Output:
        preprocessed_data
        Shape = (14, 19200)
    """

    # --------------------------------------------------------
    # Step 1: Mean removal
    # --------------------------------------------------------

    centered = (
        subject_data
        - np.mean(subject_data, axis=1, keepdims=True)
    )

    # --------------------------------------------------------
    # Step 2: Band-pass filtering
    # --------------------------------------------------------

    filtered = filtfilt(
        b,
        a,
        centered,
        axis=1
    )

    # --------------------------------------------------------
    # Step 3: Channel-wise normalization
    # --------------------------------------------------------

    mean = np.mean(
        filtered,
        axis=1,
        keepdims=True
    )

    std = np.std(
        filtered,
        axis=1,
        keepdims=True
    )

    # Prevent division by zero
    std[std == 0] = 1.0

    normalized = (
        filtered - mean
    ) / std

    return normalized


# ============================================================
# 6. TEST PREPROCESSING ON SUBJECT 1
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING TEST - SUBJECT 1")
print("=" * 70)

subject_1 = dataset[:, :, 0]

print("Original shape:", subject_1.shape)

processed_subject_1 = preprocess_subject(
    subject_1
)

print(
    "Processed shape:",
    processed_subject_1.shape
)


# ------------------------------------------------------------
# Check normalization
# ------------------------------------------------------------

print("\nChannel 1 statistics after preprocessing:")

print(
    "Mean:",
    np.mean(processed_subject_1[0])
)

print(
    "Std:",
    np.std(processed_subject_1[0])
)

print(
    "Min:",
    np.min(processed_subject_1[0])
)

print(
    "Max:",
    np.max(processed_subject_1[0])
)


# ============================================================
# 7. PLOT BEFORE / AFTER
# ============================================================

time = np.arange(num_samples) / FS

plt.figure(figsize=(12, 5))

plt.plot(
    time[:1000],
    subject_1[0, :1000]
)

plt.title(
    "STEW S01 - Channel 1 - Original"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.tight_layout()

plt.show()


plt.figure(figsize=(12, 5))

plt.plot(
    time[:1000],
    processed_subject_1[0, :1000]
)

plt.title(
    "STEW S01 - Channel 1 - Preprocessed"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Normalized Amplitude")

plt.tight_layout()

plt.show()


# ============================================================
# 8. PROCESS ALL SUBJECTS
# ============================================================

print("\n" + "=" * 70)
print("PROCESSING ALL STEW SUBJECTS")
print("=" * 70)

all_processed = []

for subject_index in range(num_subjects):

    subject_number = subject_index + 1

    subject_data = dataset[:, :, subject_index]

    processed = preprocess_subject(
        subject_data
    )

    all_processed.append(processed)

    print(
        f"Subject {subject_number:02d}/{num_subjects} processed"
    )


all_processed = np.array(all_processed)

print("\nAll preprocessed data shape:")

print(all_processed.shape)

print(
    "Expected:",
    f"({num_subjects}, {num_channels}, {num_samples})"
)


# ============================================================
# 9. WINDOWING
# ============================================================

print("\n" + "=" * 70)
print("WINDOWING")
print("=" * 70)

print("Sampling rate:", FS, "Hz")

print(
    "Window duration:",
    WINDOW_DURATION,
    "seconds"
)

print(
    "Samples per window:",
    SAMPLES_PER_WINDOW
)


windows = []
window_labels = []
window_subjects = []
window_ratings = []


for subject_index in range(num_subjects):

    subject_data = all_processed[subject_index]

    subject_class = classes[subject_index]

    # Number of complete windows
    num_windows = (
        num_samples // SAMPLES_PER_WINDOW
    )

    for window_index in range(num_windows):

        start = (
            window_index
            * SAMPLES_PER_WINDOW
        )

        end = (
            start
            + SAMPLES_PER_WINDOW
        )

        window = subject_data[:, start:end]

        # Store window
        windows.append(window)

        # Store workload class
        window_labels.append(subject_class)

        # Store subject ID
        window_subjects.append(
            subject_index + 1
        )

        # Store rating if available
        if ratings is not None:

            window_ratings.append(
                ratings[subject_index]
            )


# Convert to NumPy arrays

windows = np.array(windows)

window_labels = np.array(
    window_labels
)

window_subjects = np.array(
    window_subjects
)


if ratings is not None:

    window_ratings = np.array(
        window_ratings
    )

else:

    window_ratings = np.array([])


print("\nWindowing completed!")

print(
    "Windows shape:",
    windows.shape
)

print(
    "Labels shape:",
    window_labels.shape
)

print(
    "Subject IDs shape:",
    window_subjects.shape
)


# ============================================================
# 10. LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("WORKLOAD CLASS DISTRIBUTION")
print("=" * 70)

unique_classes, counts = np.unique(
    window_labels,
    return_counts=True
)

for cls, count in zip(
    unique_classes,
    counts
):

    print(
        f"Class {int(cls)}: {count} windows"
    )


# ============================================================
# 11. CHECK DATA QUALITY
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATA QUALITY CHECK")
print("=" * 70)

nan_count = np.isnan(
    windows
).sum()

inf_count = np.isinf(
    windows
).sum()

print("NaN values:", nan_count)

print("Inf values:", inf_count)

if nan_count == 0 and inf_count == 0:

    print(
        "Data quality check: PASSED ✓"
    )

else:

    print(
        "WARNING: Invalid values detected!"
    )


# ============================================================
# 12. SAVE PREPROCESSED DATA
# ============================================================

print("\n" + "=" * 70)
print("SAVING PREPROCESSED DATA")
print("=" * 70)

output_file = os.path.join(
    OUTPUT_DIR,
    "stew_preprocessed.npz"
)

np.savez_compressed(

    output_file,

    windows=windows,

    labels=window_labels,

    subjects=window_subjects,

    ratings=window_ratings,

    sampling_rate=FS,

    window_duration=WINDOW_DURATION,

    channels=num_channels

)


print("\nOutput directory:")

print(OUTPUT_DIR)

print("\nOutput file:")

print(output_file)


# ============================================================
# 13. VERIFY SAVED FILE
# ============================================================

if os.path.exists(output_file):

    file_size = (
        os.path.getsize(output_file)
        / (1024 ** 2)
    )

    print("\nFile saved successfully!")

    print(
        "File exists:",
        True
    )

    print(
        f"File size: {file_size:.2f} MB"
    )

else:

    print(
        "\nERROR: File was not created!"
    )


# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEW PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    "Subjects:",
    num_subjects
)

print(
    "EEG channels:",
    num_channels
)

print(
    "Sampling rate:",
    FS,
    "Hz"
)

print(
    "Window duration:",
    WINDOW_DURATION,
    "seconds"
)

print(
    "Samples per window:",
    SAMPLES_PER_WINDOW
)

print(
    "Final windows:",
    windows.shape
)

print(
    "Final labels:",
    window_labels.shape
)

print(
    "Output:",
    output_file
)

print("=" * 70)