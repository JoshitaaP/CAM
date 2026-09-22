import pickle
import numpy as np
import os
from scipy.signal import butter, filtfilt

# ============================================================
# CONFIGURATION
# ============================================================

BASE_PATH = r"D:\capstone\datasets\deap-1\deap-dataset\data_preprocessed_python"

OUTPUT_PATH = r"D:\capstone\datasets\deap-1\preprocessed"

SAMPLING_RATE = 128

# EEG frequency range
LOW_CUT = 0.5
HIGH_CUT = 45.0

# Window configuration
WINDOW_DURATION = 2
WINDOW_SAMPLES = SAMPLING_RATE * WINDOW_DURATION

# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ============================================================
# BAND-PASS FILTER
# ============================================================

def bandpass_filter(signal, lowcut, highcut, fs, order=4):

    nyquist = 0.5 * fs

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(
        order,
        [low, high],
        btype="band"
    )

    filtered_signal = filtfilt(
        b,
        a,
        signal
    )

    return filtered_signal


# ============================================================
# PREPROCESS ONE SUBJECT
# ============================================================

def preprocess_subject(subject_number):

    subject_id = f"s{subject_number:02d}"

    file_path = os.path.join(
        BASE_PATH,
        subject_id + ".dat"
    )

    print("\n" + "=" * 70)
    print(f"PROCESSING DEAP {subject_id.upper()}")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    with open(file_path, "rb") as f:
        subject = pickle.load(
            f,
            encoding="latin1"
        )

    data = subject["data"]
    labels = subject["labels"]

    print("Original data shape:", data.shape)
    print("Labels shape:", labels.shape)

    # --------------------------------------------------------
    # SELECT EEG CHANNELS
    # --------------------------------------------------------

    eeg_data = data[:, :32, :]

    print("EEG data shape:", eeg_data.shape)

    num_trials = eeg_data.shape[0]

    # --------------------------------------------------------
    # PREPROCESS ALL TRIALS
    # --------------------------------------------------------

    preprocessed_data = np.zeros_like(eeg_data)

    for trial_index in range(num_trials):

        trial = eeg_data[trial_index]

        # ----------------------------------------------------
        # MEAN REMOVAL
        # ----------------------------------------------------

        trial_centered = trial - np.mean(
            trial,
            axis=1,
            keepdims=True
        )

        # ----------------------------------------------------
        # BAND-PASS FILTER
        # ----------------------------------------------------

        filtered_trial = np.zeros_like(
            trial_centered
        )

        for channel in range(32):

            filtered_trial[channel] = bandpass_filter(
                trial_centered[channel],
                LOW_CUT,
                HIGH_CUT,
                SAMPLING_RATE
            )

        # ----------------------------------------------------
        # Z-SCORE NORMALIZATION
        # ----------------------------------------------------

        mean = np.mean(
            filtered_trial,
            axis=1,
            keepdims=True
        )

        std = np.std(
            filtered_trial,
            axis=1,
            keepdims=True
        )

        # Prevent division by zero
        std[std == 0] = 1

        normalized_trial = (
            filtered_trial - mean
        ) / std

        preprocessed_data[trial_index] = normalized_trial

        print(
            f"Trial {trial_index + 1:02d}/{num_trials} processed"
        )

    print("\nAll preprocessed data shape:")
    print(preprocessed_data.shape)

    # ========================================================
    # WINDOWING
    # ========================================================

    print("\n" + "=" * 70)
    print("WINDOWING")
    print("=" * 70)

    print("Sampling rate:", SAMPLING_RATE, "Hz")
    print("Window duration:", WINDOW_DURATION, "seconds")
    print("Samples per window:", WINDOW_SAMPLES)

    windows = []
    window_labels = []

    # --------------------------------------------------------
    # PROCESS EACH TRIAL
    # --------------------------------------------------------

    for trial_index in range(num_trials):

        trial = preprocessed_data[trial_index]

        # ----------------------------------------------------
        # LABELS
        # ----------------------------------------------------

        valence = labels[trial_index, 0]
        arousal = labels[trial_index, 1]

        # Binary classification
        # < 5  -> Low
        # >= 5 -> High

        valence_label = 0 if valence < 5 else 1
        arousal_label = 0 if arousal < 5 else 1

        # ----------------------------------------------------
        # CREATE NON-OVERLAPPING WINDOWS
        # ----------------------------------------------------

        num_windows = (
            trial.shape[1] // WINDOW_SAMPLES
        )

        for window_index in range(num_windows):

            start = (
                window_index *
                WINDOW_SAMPLES
            )

            end = (
                start +
                WINDOW_SAMPLES
            )

            window = trial[
                :,
                start:end
            ]

            # Only keep complete windows
            if window.shape[1] == WINDOW_SAMPLES:

                windows.append(window)

                window_labels.append([
                    valence_label,
                    arousal_label
                ])

    # --------------------------------------------------------
    # CONVERT TO NUMPY
    # --------------------------------------------------------

    windows = np.array(windows)

    window_labels = np.array(
        window_labels,
        dtype=np.int64
    )

    print("\nWindowing completed!")

    print("Windows shape:")
    print(windows.shape)

    print("\nWindow labels shape:")
    print(window_labels.shape)

    # ========================================================
    # LABEL DISTRIBUTION
    # ========================================================

    print("\n" + "=" * 70)
    print("LABEL DISTRIBUTION")
    print("=" * 70)

    # Valence

    valence_counts = np.bincount(
        window_labels[:, 0],
        minlength=2
    )

    print("\nValence labels:")
    print("Low:", valence_counts[0])
    print("High:", valence_counts[1])

    # Arousal

    arousal_counts = np.bincount(
        window_labels[:, 1],
        minlength=2
    )

    print("\nArousal labels:")
    print("Low:", arousal_counts[0])
    print("High:", arousal_counts[1])

    # ========================================================
    # CHECK FOR NaN / INF
    # ========================================================

    print("\n" + "=" * 70)
    print("DATA QUALITY CHECK")
    print("=" * 70)

    nan_count = np.isnan(windows).sum()
    inf_count = np.isinf(windows).sum()

    print("NaN values:", nan_count)
    print("Inf values:", inf_count)

    # ========================================================
    # SAVE
    # ========================================================

    output_file = os.path.join(
        OUTPUT_PATH,
        f"deap_preprocessed_{subject_id}.npz"
    )

    print("\n" + "=" * 70)
    print("SAVING")
    print("=" * 70)

    np.savez_compressed(
        output_file,
        windows=windows,
        labels=window_labels
    )

    print("Saved to:")
    print(output_file)

    print(
        "File exists:",
        os.path.exists(output_file)
    )

    if os.path.exists(output_file):

        file_size = (
            os.path.getsize(output_file)
            / (1024 * 1024)
        )

        print(
            f"File size: {file_size:.2f} MB"
        )

    return windows.shape, window_labels.shape


# ============================================================
# PROCESS ALL 32 DEAP SUBJECTS
# ============================================================

print("\n" + "=" * 70)
print("DEAP BATCH PREPROCESSING")
print("=" * 70)

print("Subjects: S01 - S32")
print("Output directory:")
print(OUTPUT_PATH)

successful_subjects = []
failed_subjects = []

for subject_number in range(1, 33):

    try:

        windows_shape, labels_shape = preprocess_subject(
            subject_number
        )

        successful_subjects.append(
            subject_number
        )

    except Exception as e:

        print("\n" + "!" * 70)
        print(
            f"ERROR PROCESSING S{subject_number:02d}"
        )
        print("Error:", e)
        print("!" * 70)

        failed_subjects.append(
            subject_number
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n\n" + "=" * 70)
print("DEAP BATCH PREPROCESSING COMPLETED")
print("=" * 70)

print(
    "\nSuccessfully processed:",
    len(successful_subjects),
    "/ 32"
)

print(
    "Successful subjects:",
    [
        f"S{x:02d}"
        for x in successful_subjects
    ]
)

if failed_subjects:

    print(
        "\nFailed subjects:",
        [
            f"S{x:02d}"
            for x in failed_subjects
        ]
    )

else:

    print("\nFailed subjects: None")

print("\nOutput directory:")
print(OUTPUT_PATH)

print("\nExpected files:")
print("deap_preprocessed_s01.npz")
print("deap_preprocessed_s02.npz")
print("...")
print("deap_preprocessed_s32.npz")

print("\n" + "=" * 70)
print("NEXT STEP: VERIFY ALL DEAP FILES")
print("=" * 70)