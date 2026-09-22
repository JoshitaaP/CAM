
import os
import glob
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

ANNOTATION_DIR = r"D:\capstone\datasets\28455737\Raw Dataset\Annotation"

# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("MPD-DF ANNOTATION ANALYSIS")
print("=" * 70)

# ============================================================
# FIND ANNOTATION FILES
# ============================================================

annotation_files = sorted(
    glob.glob(os.path.join(ANNOTATION_DIR, "*.txt"))
)

print("\nNumber of annotation files:", len(annotation_files))

if len(annotation_files) == 0:
    print("\nERROR: No annotation files found!")
    exit()

# ============================================================
# ANALYZE ALL ANNOTATION FILES
# ============================================================

numeric_labels = Counter()
text_annotations = Counter()

file_statistics = []

total_rows = 0
valid_numeric_rows = 0
text_rows = 0
invalid_rows = 0

# ============================================================
# PROCESS FILES
# ============================================================

for file_path in annotation_files:

    filename = os.path.basename(file_path)

    numeric_count = 0
    text_count = 0
    invalid_count = 0

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        lines = f.readlines()

    for line_number, line in enumerate(lines, start=1):

        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        total_rows += 1

        parts = [p.strip() for p in line.split(",")]

        # ----------------------------------------------------
        # Expected annotation structure
        # ----------------------------------------------------
        #
        # Example:
        #
        # 13:53:18,1,0
        #
        # or rows containing:
        #
        # time,...,Severe Artifacts
        #
        # ----------------------------------------------------

        if len(parts) < 3:
            invalid_count += 1
            invalid_rows += 1
            continue

        time_value = parts[0]

        # The annotation information may be in different columns,
        # so inspect all fields after the timestamp.

        annotation_fields = parts[1:]

        numeric_found = False
        text_found = False

        for field in annotation_fields:

            field = field.strip()

            # Try numeric label
            try:

                value = int(field)

                # We are interested in binary fatigue labels 0/1
                if value in [0, 1]:

                    numeric_labels[value] += 1

                    numeric_count += 1
                    valid_numeric_rows += 1

                    numeric_found = True

                    break

            except ValueError:
                pass

        # ----------------------------------------------------
        # If no numeric label was found, treat it as text
        # ----------------------------------------------------

        if not numeric_found:

            text_annotations_found = []

            for field in annotation_fields:

                field = field.strip()

                if field:

                    # Ignore purely numeric fields
                    try:
                        float(field)
                        continue
                    except ValueError:
                        pass

                    text_annotations[field] += 1
                    text_annotations_found.append(field)

            if text_annotations_found:

                text_count += 1
                text_rows += 1

            else:

                invalid_count += 1
                invalid_rows += 1

    file_statistics.append(
        (
            filename,
            numeric_count,
            text_count,
            invalid_count
        )
    )


# ============================================================
# NUMERIC LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("NUMERIC FATIGUE LABEL DISTRIBUTION")
print("=" * 70)

if numeric_labels:

    for label in sorted(numeric_labels):

        print(
            f"Label {label}: "
            f"{numeric_labels[label]} occurrences"
        )

else:

    print("No numeric 0/1 labels detected.")


# ============================================================
# TEXT ANNOTATION DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TEXT ANNOTATION DISTRIBUTION")
print("=" * 70)

if text_annotations:

    for annotation, count in text_annotations.most_common():

        print(
            f"{annotation}: {count} occurrences"
        )

else:

    print("No textual annotations detected.")


# ============================================================
# GLOBAL STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("GLOBAL ANNOTATION STATISTICS")
print("=" * 70)

print("Total annotation rows:", total_rows)
print("Numeric label rows:", valid_numeric_rows)
print("Text annotation rows:", text_rows)
print("Invalid/unrecognized rows:", invalid_rows)


# ============================================================
# FILE-WISE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FILE-WISE SUMMARY")
print("=" * 70)

for filename, numeric_count, text_count, invalid_count in file_statistics:

    print(
        f"{filename}: "
        f"numeric={numeric_count}, "
        f"text={text_count}, "
        f"unrecognized={invalid_count}"
    )


# ============================================================
# FIRST FILE DETAILED INSPECTION
# ============================================================

print("\n" + "=" * 70)
print("FIRST ANNOTATION FILE - DETAILED INSPECTION")
print("=" * 70)

first_file = annotation_files[0]

print("\nFile:")
print(first_file)

with open(
    first_file,
    "r",
    encoding="utf-8",
    errors="ignore"
) as f:

    first_lines = f.readlines()

print("\nAll annotation rows:")

for i, line in enumerate(first_lines, start=1):

    line = line.strip()

    if line:

        print(
            f"{i:02d}: {line}"
        )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MPD-DF ANNOTATION ANALYSIS SUMMARY")
print("=" * 70)

print("Annotation files:", len(annotation_files))
print("Total rows:", total_rows)
print("Numeric 0/1 labels:", valid_numeric_rows)
print("Text annotations:", text_rows)
print("Unrecognized rows:", invalid_rows)

print("\nNumeric labels found:")

for label in sorted(numeric_labels):

    print(
        f"  Label {label}: "
        f"{numeric_labels[label]}"
    )

print("\nText annotations found:")

for annotation, count in text_annotations.most_common():

    print(
        f"  {annotation}: "
        f"{count}"
    )

print("\nAnnotation analysis completed successfully.")