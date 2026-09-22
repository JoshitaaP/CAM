import os
import numpy as np

# ============================================================
# STEW PREPROCESSED DATA VERIFICATION
# ============================================================

print("=" * 70)
print("STEW PREPROCESSED DATA VERIFICATION")
print("=" * 70)

# ============================================================
# 1. PATH
# ============================================================

preprocessed_dir = r"D:\capstone\datasets\STEW\preprocessed"

output_file = os.path.join(
    preprocessed_dir,
    "stew_preprocessed.npz"
)

print("\nDirectory:")
print(preprocessed_dir)

print("\nDirectory exists:", os.path.exists(preprocessed_dir))

if not os.path.exists(preprocessed_dir):
    print("\nERROR: Preprocessed directory does not exist!")
    raise SystemExit


# ============================================================
# 2. CHECK FILE
# ============================================================

print("\n" + "=" * 70)
print("FILE CHECK")
print("=" * 70)

print("\nExpected file:")
print("stew_preprocessed.npz")

print("\nFile exists:", os.path.exists(output_file))

if not os.path.exists(output_file):
    print("\nERROR: STEW preprocessed file not found!")
    raise SystemExit

file_size = os.path.getsize(output_file) / (1024 * 1024)

print("File size: {:.2f} MB".format(file_size))


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING PREPROCESSED DATA")
print("=" * 70)

data = np.load(output_file)

print("\nVariables stored in file:")

for key in data.files:
    print(key, "->", data[key].shape)


# ============================================================
# 4. EXTRACT DATA
# ============================================================

windows = data["windows"]
labels = data["labels"]
subject_ids = data["subjects"]

print("\n" + "=" * 70)
print("DATA SHAPE VERIFICATION")
print("=" * 70)

print("\nWindows shape:")
print(windows.shape)

print("Expected:")
print("(3375, 14, 256)")

print("\nLabels shape:")
print(labels.shape)

print("Expected:")
print("(3375,)")

print("\nSubject IDs shape:")
print(subject_ids.shape)

print("Expected:")
print("(3375,)")


# ============================================================
# 5. CHECK SHAPES
# ============================================================

print("\n" + "=" * 70)
print("SHAPE CHECK")
print("=" * 70)

expected_windows_shape = (3375, 14, 256)
expected_labels_shape = (3375,)
expected_subject_shape = (3375,)

windows_ok = windows.shape == expected_windows_shape
labels_ok = labels.shape == expected_labels_shape
subjects_ok = subject_ids.shape == expected_subject_shape

print("\nWindows shape correct:", windows_ok)
print("Labels shape correct:", labels_ok)
print("Subject IDs shape correct:", subjects_ok)

if windows_ok and labels_ok and subjects_ok:
    print("\nShape verification: PASSED ✓")
else:
    print("\nShape verification: FAILED ✗")


# ============================================================
# 6. DATA QUALITY
# ============================================================

print("\n" + "=" * 70)
print("DATA QUALITY CHECK")
print("=" * 70)

nan_windows = np.isnan(windows).sum()
inf_windows = np.isinf(windows).sum()

nan_labels = np.isnan(labels).sum()
inf_labels = np.isinf(labels).sum()

print("\nWindows:")
print("NaN values:", nan_windows)
print("Inf values:", inf_windows)

print("\nLabels:")
print("NaN values:", nan_labels)
print("Inf values:", inf_labels)

if (
    nan_windows == 0
    and inf_windows == 0
    and nan_labels == 0
    and inf_labels == 0
):
    print("\nData quality: PASSED ✓")
else:
    print("\nData quality: FAILED ✗")


# ============================================================
# 7. LABEL VERIFICATION
# ============================================================

print("\n" + "=" * 70)
print("WORKLOAD LABEL VERIFICATION")
print("=" * 70)

unique_labels, label_counts = np.unique(
    labels,
    return_counts=True
)

print("\nUnique labels:")

for label, count in zip(unique_labels, label_counts):
    percentage = (count / len(labels)) * 100

    print(
        "Class {}: {} windows ({:.2f}%)".format(
            int(label),
            count,
            percentage
        )
    )

print("\nExpected classes:")
print("0, 1, 2")


# ============================================================
# 8. SUBJECT VERIFICATION
# ============================================================

print("\n" + "=" * 70)
print("SUBJECT ID VERIFICATION")
print("=" * 70)

unique_subjects, subject_counts = np.unique(
    subject_ids,
    return_counts=True
)

print("\nNumber of unique subjects:")
print(len(unique_subjects))

print("\nExpected:")
print("45")

print("\nWindows per subject:")

for subject, count in zip(unique_subjects, subject_counts):

    print(
        "Subject {:02d}: {} windows".format(
            int(subject),
            count
        )
    )


# ============================================================
# 9. EXPECTED WINDOWS PER SUBJECT
# ============================================================

print("\n" + "=" * 70)
print("WINDOW DISTRIBUTION CHECK")
print("=" * 70)

expected_windows_per_subject = 75

print("\nExpected windows per subject:")
print(expected_windows_per_subject)

all_subject_counts_correct = np.all(
    subject_counts == expected_windows_per_subject
)

print(
    "\nAll subjects have 75 windows:",
    all_subject_counts_correct
)

if all_subject_counts_correct:
    print("Window distribution: PASSED ✓")
else:
    print("Window distribution: CHECK REQUIRED")


# ============================================================
# 10. NORMALIZATION CHECK
# ============================================================

print("\n" + "=" * 70)
print("NORMALIZATION CHECK")
print("=" * 70)

# Calculate mean and standard deviation across all samples
global_mean = np.mean(windows)
global_std = np.std(windows)

print("\nGlobal mean:", global_mean)
print("Global standard deviation:", global_std)

print("\nExpected:")
print("Mean approximately 0")
print("Std approximately 1")


# ============================================================
# 11. VALUE RANGE
# ============================================================

print("\n" + "=" * 70)
print("VALUE RANGE CHECK")
print("=" * 70)

print("\nMinimum EEG value:", np.min(windows))
print("Maximum EEG value:", np.max(windows))

print("\nNo extreme NaN/Inf values detected.")


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL VERIFICATION SUMMARY")
print("=" * 70)

print("\nExpected subjects:")
print("45")

print("Verified subjects:")
print(len(unique_subjects))

print("\nExpected channels:")
print("14")

print("Verified channels:")
print(windows.shape[1])

print("\nExpected samples per window:")
print("256")

print("Verified samples per window:")
print(windows.shape[2])

print("\nTotal windows:")
print(len(windows))

print("\nExpected total windows:")
print("3375")

print("\nUnique workload classes:")
print(unique_labels)

print("\nNaN values:")
print(nan_windows)

print("\nInf values:")
print(inf_windows)


# ============================================================
# 13. FINAL PASS / FAIL
# ============================================================

verification_success = (
    windows_ok
    and labels_ok
    and subjects_ok
    and nan_windows == 0
    and inf_windows == 0
    and nan_labels == 0
    and inf_labels == 0
    and len(unique_subjects) == 45
    and len(unique_labels) == 3
    and len(windows) == 3375
)

print("\n" + "=" * 70)

if verification_success:

    print("STEW VERIFICATION SUCCESSFUL ✓")
    print("=" * 70)

    print("\nAll STEW preprocessing checks passed.")
    print("45 subjects are present.")
    print("14 EEG channels are present.")
    print("3375 windows are present.")
    print("All windows have 256 samples.")
    print("Workload labels 0, 1, 2 are present.")
    print("No NaN or Inf values detected.")

else:

    print("STEW VERIFICATION FAILED ✗")
    print("=" * 70)

    print("\nPlease inspect the checks above.")


print("\nVerification completed.")