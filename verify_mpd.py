import os
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

PREPROCESSED_FILE = (
    r"D:\capstone\datasets\28455737\preprocessed"
    r"\mpd_preprocessed.npz"
)

EXPECTED_SUBJECTS = 50
EXPECTED_CHANNELS = 32
EXPECTED_SAMPLES_PER_WINDOW = 1000


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("MPD-DF PREPROCESSED DATA VERIFICATION")
print("=" * 70)


# ============================================================
# FILE CHECK
# ============================================================

print("\n" + "=" * 70)
print("FILE CHECK")
print("=" * 70)

print("\nExpected file:")
print("mpd_preprocessed.npz")

print("\nFile exists:")
print(os.path.exists(PREPROCESSED_FILE))

if not os.path.exists(PREPROCESSED_FILE):
    print("\nERROR: Preprocessed file not found!")
    exit()

file_size_mb = os.path.getsize(PREPROCESSED_FILE) / (1024 * 1024)

print("File size:")
print(f"{file_size_mb:.2f} MB")


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING PREPROCESSED DATA")
print("=" * 70)

data = np.load(PREPROCESSED_FILE)

print("\nVariables stored in file:")

for key in data.files:
    print(f"{key} -> {data[key].shape}")


# ============================================================
# CHECK REQUIRED VARIABLES
# ============================================================

required_variables = [
    "windows",
    "labels",
    "subjects"
]

print("\n" + "=" * 70)
print("VARIABLE CHECK")
print("=" * 70)

all_variables_present = True

for variable in required_variables:

    exists = variable in data.files

    print(f"{variable}: {exists}")

    if not exists:
        all_variables_present = False
        print(
            f"\nERROR: Required variable '{variable}' "
            "is missing!"
        )

if not all_variables_present:
    print("\nVariable verification: FAILED ✗")
    exit()

print("\nAll required variables are present: PASSED ✓")


# ============================================================
# LOAD VARIABLES
# ============================================================

windows = data["windows"]
labels = data["labels"]
subjects = data["subjects"]


# ============================================================
# DATA SHAPE VERIFICATION
# ============================================================

print("\n" + "=" * 70)
print("DATA SHAPE VERIFICATION")
print("=" * 70)

print("\nWindows shape:")
print(windows.shape)

print("\nLabels shape:")
print(labels.shape)

print("\nSubjects shape:")
print(subjects.shape)

num_windows = windows.shape[0]


# ============================================================
# SHAPE CHECK
# ============================================================

print("\n" + "=" * 70)
print("SHAPE CHECK")
print("=" * 70)

windows_shape_correct = (
    windows.ndim == 3
    and windows.shape[1] == EXPECTED_CHANNELS
    and windows.shape[2] == EXPECTED_SAMPLES_PER_WINDOW
)

labels_shape_correct = (
    labels.ndim == 1
    and labels.shape[0] == num_windows
)

subjects_shape_correct = (
    subjects.ndim == 1
    and subjects.shape[0] == num_windows
)

print(
    "Windows shape correct:",
    windows_shape_correct
)

print(
    "Labels shape correct:",
    labels_shape_correct
)

print(
    "Subjects shape correct:",
    subjects_shape_correct
)

if (
    windows_shape_correct
    and labels_shape_correct
    and subjects_shape_correct
):

    print("\nShape verification: PASSED ✓")

else:

    print("\nShape verification: FAILED ✗")


# ============================================================
# DATA QUALITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("DATA QUALITY CHECK")
print("=" * 70)


# ------------------------------------------------------------
# WINDOWS
# ------------------------------------------------------------

print("\nWindows:")

window_nan = np.isnan(windows).sum()
window_inf = np.isinf(windows).sum()

print("NaN values:", window_nan)
print("Inf values:", window_inf)


# ------------------------------------------------------------
# LABELS
# ------------------------------------------------------------

print("\nLabels:")

label_nan = np.isnan(labels).sum()
label_inf = np.isinf(labels).sum()

print("NaN values:", label_nan)
print("Inf values:", label_inf)


# ------------------------------------------------------------
# SUBJECT IDs
# ------------------------------------------------------------

print("\nSubjects:")

# Subject IDs are identifiers, not continuous numerical EEG data.
# Therefore NaN/Inf checks are not appropriate here.

if subjects.dtype.kind in "OUS":
    subject_invalid = np.sum(
        np.array([
            str(x).strip() == ""
            for x in subjects
        ])
    )

    print("Subject ID type:", subjects.dtype)
    print("Empty subject IDs:", subject_invalid)

else:

    print("Subject ID type:", subjects.dtype)
    print("Subject IDs are valid identifier values.")
    print("NaN/Inf check: Not applicable")


# ============================================================
# LABEL VERIFICATION
# ============================================================

print("\n" + "=" * 70)
print("LABEL VERIFICATION")
print("=" * 70)

unique_labels, label_counts = np.unique(
    labels,
    return_counts=True
)

print("\nUnique labels:")

for label, count in zip(
    unique_labels,
    label_counts
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

print("\nExpected:")
print(EXPECTED_SUBJECTS)

print("\nSubject window distribution:")

for subject, count in zip(
    unique_subjects,
    subject_counts
):

    print(
        f"Subject {int(subject):02d}: "
        f"{count} windows"
    )


# ============================================================
# SUBJECT COUNT CHECK
# ============================================================

subject_count_correct = (
    len(unique_subjects) == EXPECTED_SUBJECTS
)

print("\nSubject count correct:")
print(subject_count_correct)


# ============================================================
# WINDOW DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("WINDOW DISTRIBUTION CHECK")
print("=" * 70)

print("\nMinimum windows per subject:")
print(np.min(subject_counts))

print("\nMaximum windows per subject:")
print(np.max(subject_counts))

print("\nMean windows per subject:")
print(np.mean(subject_counts))

print("\nSubjects with windows:")
print(len(unique_subjects))


# ============================================================
# EEG VALUE RANGE
# ============================================================

print("\n" + "=" * 70)
print("EEG VALUE RANGE CHECK")
print("=" * 70)

print("\nMinimum EEG value:")
print(np.min(windows))

print("\nMaximum EEG value:")
print(np.max(windows))

print("\nMean EEG value:")
print(np.mean(windows))

print("\nStandard deviation:")
print(np.std(windows))


# ============================================================
# FINAL VERIFICATION SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL VERIFICATION SUMMARY")
print("=" * 70)

print("\nExpected subjects:")
print(EXPECTED_SUBJECTS)

print("Verified subjects:")
print(len(unique_subjects))

print("\nEEG channels:")
print(windows.shape[1])

print("Expected channels:")
print(EXPECTED_CHANNELS)

print("\nSamples per window:")
print(windows.shape[2])

print("Expected samples per window:")
print(EXPECTED_SAMPLES_PER_WINDOW)

print("\nTotal windows:")
print(len(windows))

print("\nUnique labels:")
print(unique_labels)

print("\nWindow NaN values:")
print(window_nan)

print("\nWindow Inf values:")
print(window_inf)

print("\nLabel NaN values:")
print(label_nan)

print("\nLabel Inf values:")
print(label_inf)


# ============================================================
# FINAL RESULT
# ============================================================

verification_passed = (
    len(unique_subjects) == EXPECTED_SUBJECTS
    and windows.shape[1] == EXPECTED_CHANNELS
    and windows.shape[2] == EXPECTED_SAMPLES_PER_WINDOW
    and labels.shape[0] == windows.shape[0]
    and subjects.shape[0] == windows.shape[0]
    and window_nan == 0
    and window_inf == 0
    and label_nan == 0
    and label_inf == 0
)


print("\n" + "=" * 70)

if verification_passed:

    print("MPD-DF VERIFICATION SUCCESSFUL ✓")
    print("=" * 70)

    print("\nAll structural and data-quality checks passed.")

    print(
        f"{len(unique_subjects)} subjects are present."
    )

    print(
        f"{windows.shape[1]} EEG channels are present."
    )

    print(
        f"{len(windows)} windows are present."
    )

    print(
        f"All windows have {windows.shape[2]} samples."
    )

    print(
        f"Labels present: {unique_labels}"
    )

    print(
        "No NaN or Inf values detected in EEG windows or labels."
    )

else:

    print("MPD-DF VERIFICATION FAILED ✗")
    print("=" * 70)

print("\nVerification completed.")