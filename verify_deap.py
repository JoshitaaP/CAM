import os
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

PREPROCESSED_DIR = r"D:\capstone\datasets\deap-1\preprocessed"

EXPECTED_SUBJECTS = 32
EXPECTED_WINDOWS = 1240
EXPECTED_CHANNELS = 32
EXPECTED_SAMPLES = 256

# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("DEAP PREPROCESSED DATA VERIFICATION")
print("=" * 70)

print("\nDirectory:")
print(PREPROCESSED_DIR)

# ============================================================
# CHECK DIRECTORY
# ============================================================

if not os.path.exists(PREPROCESSED_DIR):
    print("\nERROR: Preprocessed directory does not exist!")
    print("Check the path:")
    print(PREPROCESSED_DIR)
    exit()

print("\nDirectory exists: True")

# ============================================================
# FIND FILES
# ============================================================

files = sorted(
    [
        f for f in os.listdir(PREPROCESSED_DIR)
        if f.startswith("deap_preprocessed_s")
        and f.endswith(".npz")
    ]
)

print("\nFiles found:", len(files))
print("Expected:", EXPECTED_SUBJECTS)

# ============================================================
# CHECK ALL SUBJECTS
# ============================================================

missing_subjects = []

for subject in range(1, EXPECTED_SUBJECTS + 1):

    filename = f"deap_preprocessed_s{subject:02d}.npz"

    if filename not in files:
        missing_subjects.append(subject)

# ============================================================
# PRINT FILE STATUS
# ============================================================

print("\n" + "=" * 70)
print("FILE CHECK")
print("=" * 70)

if len(missing_subjects) == 0:

    print("All 32 subject files are present! ✓")

else:

    print("Missing subjects:", missing_subjects)

# ============================================================
# VERIFY DATA SHAPES
# ============================================================

print("\n" + "=" * 70)
print("DATA SHAPE VERIFICATION")
print("=" * 70)

failed_subjects = []

total_windows = 0

for subject in range(1, EXPECTED_SUBJECTS + 1):

    filename = f"deap_preprocessed_s{subject:02d}.npz"
    filepath = os.path.join(PREPROCESSED_DIR, filename)

    if not os.path.exists(filepath):
        continue

    try:

        data = np.load(filepath)

        # Show available arrays
        arrays = data.files

        # ----------------------------------------------------
        # Find windows and labels
        # ----------------------------------------------------

        if "windows" in arrays:
            windows = data["windows"]
        else:
            print(f"\nWARNING: {filename} does not contain 'windows'")
            print("Available arrays:", arrays)
            failed_subjects.append(subject)
            continue

        if "labels" in arrays:
            labels = data["labels"]
        else:
            print(f"\nWARNING: {filename} does not contain 'labels'")
            print("Available arrays:", arrays)
            failed_subjects.append(subject)
            continue

        # ----------------------------------------------------
        # Shape check
        # ----------------------------------------------------

        expected_window_shape = (
            EXPECTED_WINDOWS,
            EXPECTED_CHANNELS,
            EXPECTED_SAMPLES
        )

        expected_label_shape = (
            EXPECTED_WINDOWS,
            2
        )

        shape_correct = (
            windows.shape == expected_window_shape
            and labels.shape == expected_label_shape
        )

        # ----------------------------------------------------
        # NaN / Inf check
        # ----------------------------------------------------

        has_nan = np.isnan(windows).any()
        has_inf = np.isinf(windows).any()

        values_correct = not has_nan and not has_inf

        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        print(
            f"S{subject:02d}: "
            f"windows={windows.shape}, "
            f"labels={labels.shape}, "
            f"NaN={has_nan}, "
            f"Inf={has_inf}"
        )

        # ----------------------------------------------------
        # Record failures
        # ----------------------------------------------------

        if not shape_correct or not values_correct:
            failed_subjects.append(subject)

        total_windows += windows.shape[0]

        data.close()

    except Exception as e:

        print(f"S{subject:02d}: ERROR -> {e}")
        failed_subjects.append(subject)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL VERIFICATION SUMMARY")
print("=" * 70)

print("\nExpected subjects:", EXPECTED_SUBJECTS)
print("Files found:", len(files))

print("\nExpected shape per subject:")
print(
    f"Windows: ({EXPECTED_WINDOWS}, "
    f"{EXPECTED_CHANNELS}, "
    f"{EXPECTED_SAMPLES})"
)

print("Labels:", (EXPECTED_WINDOWS, 2))

print("\nTotal windows across verified subjects:", total_windows)

print("\nMissing subjects:", missing_subjects)
print("Failed subjects:", failed_subjects)

# ============================================================
# FINAL RESULT
# ============================================================

if (
    len(files) == EXPECTED_SUBJECTS
    and len(missing_subjects) == 0
    and len(failed_subjects) == 0
):

    print("\n" + "=" * 70)
    print("DEAP VERIFICATION SUCCESSFUL ✓")
    print("=" * 70)

    print("\nAll 32 subjects are correctly preprocessed.")
    print("All expected windows and labels are present.")
    print("No NaN or Inf values detected.")

else:

    print("\n" + "=" * 70)
    print("DEAP VERIFICATION FAILED")
    print("=" * 70)

    print("\nPlease check the missing/failed subjects above.")

print("\nVerification completed.")