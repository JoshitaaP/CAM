import os
import glob
import numpy as np
import mne

# ============================================================
# CONFIGURATION
# ============================================================

EEG_DIR = r"D:\capstone\datasets\28455737\Raw Dataset\EEG"
ANNOTATION_DIR = r"D:\capstone\datasets\28455737\Raw Dataset\Annotation"

print("=" * 70)
print("MPD-DF DATASET - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# ============================================================
# FIND FILES
# ============================================================

eeg_files = sorted(glob.glob(os.path.join(EEG_DIR, "*.edf")))
annotation_files = sorted(
    glob.glob(os.path.join(ANNOTATION_DIR, "*.txt"))
)

print("\nEEG directory:")
print(EEG_DIR)

print("\nAnnotation directory:")
print(ANNOTATION_DIR)

print("\nNumber of EEG files:", len(eeg_files))
print("Number of annotation files:", len(annotation_files))

# ============================================================
# CHECK FILE PAIRING
# ============================================================

print("\n" + "=" * 70)
print("FILE PAIRING CHECK")
print("=" * 70)

for eeg_file in eeg_files[:10]:
    name = os.path.basename(eeg_file)
    subject_id = name.replace("MPDDF_raw_", "").replace("_EEG.edf", "")

    expected_annotation = os.path.join(
        ANNOTATION_DIR,
        f"MPDDF_raw_{subject_id}_Annotation.txt"
    )

    print(
        f"{name} -> "
        f"{os.path.exists(expected_annotation)}"
    )

# ============================================================
# INSPECT FIRST EEG FILE
# ============================================================

print("\n" + "=" * 70)
print("FIRST EEG FILE INSPECTION")
print("=" * 70)

first_file = eeg_files[0]

print("\nFile:")
print(first_file)

raw = mne.io.read_raw_edf(
    first_file,
    preload=False,
    verbose=False
)

print("\nChannel names:")
for i, ch in enumerate(raw.ch_names):
    print(f"{i + 1:02d}: {ch}")

print("\nNumber of channels:")
print(len(raw.ch_names))

print("\nSampling frequency:")
print(raw.info["sfreq"], "Hz")

print("\nNumber of samples:")
print(raw.n_times)

print("\nRecording duration:")
print(raw.times[-1], "seconds")

print("\nData type:")
print(raw.get_data(
    start=0,
    stop=min(1000, raw.n_times)
).dtype)

# ============================================================
# SAMPLE EEG DATA
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE EEG DATA")
print("=" * 70)

sample_data = raw.get_data(
    start=0,
    stop=min(1000, raw.n_times)
)

print("\nSample shape:")
print(sample_data.shape)

print("\nNaN values:")
print(np.isnan(sample_data).sum())

print("\nInf values:")
print(np.isinf(sample_data).sum())

print("\nMinimum:")
print(np.min(sample_data))

print("\nMaximum:")
print(np.max(sample_data))

print("\nMean:")
print(np.mean(sample_data))

print("\nStandard deviation:")
print(np.std(sample_data))

# ============================================================
# INSPECT FIRST ANNOTATION FILE
# ============================================================

print("\n" + "=" * 70)
print("FIRST ANNOTATION FILE")
print("=" * 70)

first_annotation = annotation_files[0]

print("\nFile:")
print(first_annotation)

with open(
    first_annotation,
    "r",
    encoding="utf-8",
    errors="ignore"
) as f:
    lines = [line.strip() for line in f if line.strip()]

print("\nNumber of annotation lines:")
print(len(lines))

print("\nFirst 20 annotation lines:")

for line in lines[:20]:
    print(line)

# ============================================================
# PARSE ANNOTATION LABELS
# ============================================================

print("\n" + "=" * 70)
print("ANNOTATION LABEL ANALYSIS")
print("=" * 70)

labels = []

for line in lines:

    parts = line.split(",")

    if len(parts) >= 3:

        try:
            label = int(parts[2])
            labels.append(label)

        except ValueError:
            pass

if labels:

    unique, counts = np.unique(
        labels,
        return_counts=True
    )

    print("\nUnique labels:")

    for label, count in zip(unique, counts):
        print(
            f"Label {label}: "
            f"{count} occurrences"
        )

else:

    print("No numeric labels found.")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MPD-DF EDA SUMMARY")
print("=" * 70)

print("\nEEG files:")
print(len(eeg_files))

print("Annotation files:")
print(len(annotation_files))

print("Channels:")
print(len(raw.ch_names))

print("Sampling rate:")
print(raw.info["sfreq"], "Hz")

print("Recording duration:")
print(round(raw.times[-1], 2), "seconds")

print("Annotation lines in first subject:")
print(len(lines))

print("\nEDA completed successfully.")