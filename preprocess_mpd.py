import os
import glob
import numpy as np
import mne
from scipy.signal import butter, sosfiltfilt

# ============================================================
# CONFIGURATION
# ============================================================

EEG_DIR = r"D:\capstone\datasets\28455737\Raw Dataset\EEG"
ANNOTATION_DIR = r"D:\capstone\datasets\28455737\Raw Dataset\Annotation"

OUTPUT_DIR = r"D:\capstone\datasets\28455737\preprocessed"

SAMPLING_RATE = 500

WINDOW_DURATION = 2
WINDOW_SAMPLES = SAMPLING_RATE * WINDOW_DURATION

LOW_CUTOFF = 0.3
HIGH_CUTOFF = 35.0

FILTER_ORDER = 4

# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("MPD-DF EEG PREPROCESSING")
print("=" * 70)

# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# FIND FILES
# ============================================================

eeg_files = sorted(
    glob.glob(os.path.join(EEG_DIR, "*.edf"))
)

annotation_files = sorted(
    glob.glob(os.path.join(ANNOTATION_DIR, "*.txt"))
)

print("\nEEG files:", len(eeg_files))
print("Annotation files:", len(annotation_files))

if len(eeg_files) == 0:
    raise FileNotFoundError("No EEG files found.")

if len(annotation_files) == 0:
    raise FileNotFoundError("No annotation files found.")

# ============================================================
# CREATE ANNOTATION LOOKUP
# ============================================================

annotation_lookup = {}

for annotation_file in annotation_files:

    filename = os.path.basename(annotation_file)

    subject_id = (
        filename
        .replace("MPDDF_raw_", "")
        .replace("_Annotation.txt", "")
    )

    annotation_lookup[subject_id] = annotation_file

# ============================================================
# FILTER DESIGN
# ============================================================

print("\n" + "=" * 70)
print("FILTER DESIGN")
print("=" * 70)

print("Sampling rate:", SAMPLING_RATE, "Hz")
print("Band-pass:", LOW_CUTOFF, "-", HIGH_CUTOFF, "Hz")
print("Filter order:", FILTER_ORDER)

sos = butter(
    FILTER_ORDER,
    [LOW_CUTOFF, HIGH_CUTOFF],
    btype="bandpass",
    fs=SAMPLING_RATE,
    output="sos"
)

# ============================================================
# LABEL CONVERSION
# ============================================================

def convert_label(raw_label):

    raw_label = int(raw_label)

    # --------------------------------------------------------
    # IMPORTANT:
    # Preserve original annotation labels for now.
    #
    # We are NOT assuming what 0/1/2/3 mean.
    # --------------------------------------------------------

    if raw_label in [0, 1, 2, 3, 4]:
        return raw_label

    return None


# ============================================================
# PROCESS ANNOTATION FILE
# ============================================================

def read_annotations(annotation_file):

    annotations = []

    with open(
        annotation_file,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        lines = f.readlines()

    for line in lines:

        line = line.strip()

        if not line:
            continue

        parts = [
            p.strip()
            for p in line.split(",")
        ]

        if len(parts) < 3:
            continue

        timestamp = parts[0]

        try:
            segment_number = int(parts[1])
            raw_label = int(parts[2])
        except ValueError:
            continue

        label = convert_label(raw_label)

        if label is None:
            continue

        annotations.append(
            {
                "timestamp": timestamp,
                "segment": segment_number,
                "label": label
            }
        )

    return annotations


# ============================================================
# GLOBAL STORAGE
# ============================================================

all_windows = []
all_labels = []
all_subjects = []

# ============================================================
# PROCESS SUBJECTS
# ============================================================

print("\n" + "=" * 70)
print("PROCESSING MPD-DF SUBJECTS")
print("=" * 70)

for subject_number, eeg_file in enumerate(eeg_files, start=1):

    filename = os.path.basename(eeg_file)

    subject_id = (
        filename
        .replace("MPDDF_raw_", "")
        .replace("_EEG.edf", "")
    )

    print(
        f"\nSubject {subject_number}/{len(eeg_files)} "
        f"({subject_id})"
    )

    # --------------------------------------------------------
    # FIND ANNOTATION FILE
    # --------------------------------------------------------

    if subject_id not in annotation_lookup:

        print("WARNING: Annotation file not found.")
        continue

    annotation_file = annotation_lookup[subject_id]

    # --------------------------------------------------------
    # READ ANNOTATIONS
    # --------------------------------------------------------

    annotations = read_annotations(
        annotation_file
    )

    print(
        "Valid annotation rows:",
        len(annotations)
    )

    if len(annotations) == 0:

        print(
            "WARNING: No valid annotations."
        )

        continue

    # --------------------------------------------------------
    # LOAD EEG
    # --------------------------------------------------------

    raw = mne.io.read_raw_edf(
        eeg_file,
        preload=True,
        verbose=False
    )

    eeg = raw.get_data()

    print(
        "Original EEG shape:",
        eeg.shape
    )

    # --------------------------------------------------------
    # VERIFY SAMPLING RATE
    # --------------------------------------------------------

    actual_sampling_rate = raw.info["sfreq"]

    if actual_sampling_rate != SAMPLING_RATE:

        print(
            "WARNING: Expected sampling rate:",
            SAMPLING_RATE
        )

        print(
            "Actual sampling rate:",
            actual_sampling_rate
        )

    # --------------------------------------------------------
    # REMOVE NON-EEG CHANNELS IF ANY
    # --------------------------------------------------------

    eeg = eeg.astype(np.float64)

    # --------------------------------------------------------
    # FILTER EACH CHANNEL
    # --------------------------------------------------------

    print("Applying band-pass filter...")

    filtered_eeg = np.zeros_like(eeg)

    for channel in range(eeg.shape[0]):

        filtered_eeg[channel] = sosfiltfilt(
            sos,
            eeg[channel]
        )

    # --------------------------------------------------------
    # CHANNEL-WISE NORMALIZATION
    # --------------------------------------------------------

    print(
        "Applying channel-wise normalization..."
    )

    for channel in range(filtered_eeg.shape[0]):

        mean = np.mean(
            filtered_eeg[channel]
        )

        std = np.std(
            filtered_eeg[channel]
        )

        if std > 0:

            filtered_eeg[channel] = (
                filtered_eeg[channel] - mean
            ) / std

    # --------------------------------------------------------
    # CREATE WINDOWS FROM ANNOTATED SEGMENTS
    # --------------------------------------------------------

    subject_windows = []
    subject_labels = []

    for annotation in annotations:

        segment_number = annotation["segment"]
        label = annotation["label"]

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # MPD-DF annotations identify 30-second segments.
        #
        # Segment 1 begins at sample 0.
        # Segment N begins at:
        #
        # (N - 1) * 30 seconds
        # ----------------------------------------------------

        segment_start = (
            (segment_number - 1)
            * 30
            * SAMPLING_RATE
        )

        segment_end = (
            segment_start
            + 30
            * SAMPLING_RATE
        )

        # ----------------------------------------------------
        # CHECK BOUNDS
        # ----------------------------------------------------

        if segment_start >= filtered_eeg.shape[1]:

            continue

        segment_end = min(
            segment_end,
            filtered_eeg.shape[1]
        )

        segment = filtered_eeg[
            :,
            segment_start:segment_end
        ]

        # ----------------------------------------------------
        # SPLIT 30 SECOND SEGMENT
        # INTO 2 SECOND WINDOWS
        # ----------------------------------------------------

        number_of_windows = (
            segment.shape[1]
            // WINDOW_SAMPLES
        )

        for window_index in range(
            number_of_windows
        ):

            start = (
                window_index
                * WINDOW_SAMPLES
            )

            end = (
                start
                + WINDOW_SAMPLES
            )

            window = segment[
                :,
                start:end
            ]

            if window.shape[1] != WINDOW_SAMPLES:
                continue

            if (
                np.isnan(window).any()
                or np.isinf(window).any()
            ):
                continue

            subject_windows.append(
                window.astype(np.float32)
            )

            subject_labels.append(
                label
            )

    # --------------------------------------------------------
    # STORE SUBJECT DATA
    # --------------------------------------------------------

    print(
        "Generated windows:",
        len(subject_windows)
    )

    all_windows.extend(
        subject_windows
    )

    all_labels.extend(
        subject_labels
    )

    all_subjects.extend(
        [subject_id] * len(subject_windows)
    )


# ============================================================
# CONVERT TO NUMPY ARRAYS
# ============================================================

print("\n" + "=" * 70)
print("CREATING FINAL DATASET")
print("=" * 70)

windows = np.array(
    all_windows,
    dtype=np.float32
)

labels = np.array(
    all_labels,
    dtype=np.int64
)

subjects = np.array(
    all_subjects
)

print("\nWindows shape:")
print(windows.shape)

print("\nLabels shape:")
print(labels.shape)

print("\nSubjects shape:")
print(subjects.shape)

# ============================================================
# DATA QUALITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATA QUALITY CHECK")
print("=" * 70)

print(
    "NaN values:",
    np.isnan(windows).sum()
)

print(
    "Inf values:",
    np.isinf(windows).sum()
)

if (
    np.isnan(windows).any()
    or np.isinf(windows).any()
):

    raise ValueError(
        "NaN or Inf values detected!"
    )

print(
    "Data quality: PASSED ✓"
)

# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)

unique_labels, counts = np.unique(
    labels,
    return_counts=True
)

for label, count in zip(
    unique_labels,
    counts
):

    percentage = (
        count / len(labels)
    ) * 100

    print(
        f"Label {label}: "
        f"{count} windows "
        f"({percentage:.2f}%)"
    )

# ============================================================
# SUBJECT DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("SUBJECT DISTRIBUTION")
print("=" * 70)

unique_subjects, subject_counts = np.unique(
    subjects,
    return_counts=True
)

for subject, count in zip(
    unique_subjects,
    subject_counts
):

    print(
        f"Subject {subject}: "
        f"{count} windows"
    )

# ============================================================
# SAVE DATASET
# ============================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "mpd_preprocessed.npz"
)

print("\n" + "=" * 70)
print("SAVING PREPROCESSED DATA")
print("=" * 70)

np.savez_compressed(
    output_file,
    windows=windows,
    labels=labels,
    subjects=subjects,
    sampling_rate=SAMPLING_RATE,
    window_duration=WINDOW_DURATION,
    channels=windows.shape[1],
    samples_per_window=WINDOW_SAMPLES
)

print("\nOutput file:")
print(output_file)

print(
    "File exists:",
    os.path.exists(output_file)
)

if os.path.exists(output_file):

    size_mb = (
        os.path.getsize(output_file)
        / (1024 ** 2)
    )

    print(
        f"File size: {size_mb:.2f} MB"
    )

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MPD-DF PREPROCESSING COMPLETED")
print("=" * 70)

print("Subjects:", len(unique_subjects))
print("Channels:", windows.shape[1])
print("Sampling rate:", SAMPLING_RATE, "Hz")
print(
    "Window duration:",
    WINDOW_DURATION,
    "seconds"
)

print(
    "Samples per window:",
    WINDOW_SAMPLES
)

print(
    "Total windows:",
    len(windows)
)

print(
    "Labels:",
    unique_labels
)

print(
    "Output:",
    output_file
)

print("=" * 70)