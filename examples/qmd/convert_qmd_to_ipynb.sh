#!/usr/bin/env bash

# Production script to convert all .qmd files to executed .ipynb notebooks
# Outputs to same directory as source .qmd files, with standard naming

# Configuration
BASE_DIR="."
FOLDERS=("basics" "data" "eda" "model" "multivariate" "state")

# Counters
total_files=0
converted=0
failed=0
declare -a failed_files

# Log file
LOG_FILE="conversion_$(date +%Y%m%d_%H%M%S).log"

echo "========================================================"
echo "QMD to IPYNB Conversion (Production)"
echo "========================================================"
echo "Base directory: $BASE_DIR"
echo "Folders: ${FOLDERS[*]}"
echo "Log file: $LOG_FILE"
echo "========================================================"
echo ""

# Function to convert a single file
convert_file() {
  local qmd_file=$1
  local dir=$(dirname "$qmd_file")
  local filename=$(basename "$qmd_file" .qmd)
  local output_file="$dir/${filename}.ipynb"

  echo "Converting: $qmd_file" | tee -a "$LOG_FILE"
  echo "  Output: $output_file" | tee -a "$LOG_FILE"

  # Step 1: Convert .qmd to .ipynb using quarto (no execution)
  echo "  [Step 1] Converting to .ipynb format..." | tee -a "$LOG_FILE"
  if quarto render "$qmd_file" --to ipynb --no-execute 2>&1 | tee -a "$LOG_FILE"; then

    if [ ! -f "$output_file" ]; then
      echo "  ✗ Failed: Notebook file not found at $output_file" | tee -a "$LOG_FILE"
      return 1
    fi

    # Step 2: Execute the notebook using jupyter nbconvert
    echo "  [Step 2] Executing notebook with jupyter..." | tee -a "$LOG_FILE"
    if jupyter nbconvert --to notebook --execute --inplace "$output_file" 2>&1 | tee -a "$LOG_FILE"; then
      echo "  ✓ Success: Converted and executed" | tee -a "$LOG_FILE"
      return 0
    else
      echo "  ✗ Failed during execution" | tee -a "$LOG_FILE"
      return 1
    fi
  else
    echo "  ✗ Failed during conversion" | tee -a "$LOG_FILE"
    return 1
  fi
}

# Process each folder
for folder in "${FOLDERS[@]}"; do
  folder_path="$BASE_DIR/$folder"

  if [ ! -d "$folder_path" ]; then
    echo "Warning: Folder not found: $folder_path" | tee -a "$LOG_FILE"
    continue
  fi

  echo "========================================================"
  echo "Processing folder: $folder"
  echo "========================================================"

  folder_count=0
  folder_failed=0

  # Find and process all .qmd files in folder
  while IFS= read -r qmd_file; do
    if [ -z "$qmd_file" ]; then
      continue
    fi

    ((total_files++))
    ((folder_count++))

    if convert_file "$qmd_file"; then
      ((converted++))
    else
      ((failed++))
      ((folder_failed++))
      failed_files+=("$qmd_file")
    fi
    echo ""
  done < <(find "$folder_path" -maxdepth 1 -name "*.qmd" -type f | sort)

  echo "Folder summary: $folder_count total, $((folder_count - folder_failed)) succeeded, $folder_failed failed"
  echo ""
done

# Final Summary
echo "========================================================"
echo "Final Conversion Summary:"
echo "========================================================"
echo "Total files processed: $total_files"
echo "Successfully converted: $converted"
echo "Failed: $failed"
echo "========================================================"

if [ $failed -gt 0 ]; then
  echo ""
  echo "Failed files:"
  for file in "${failed_files[@]}"; do
    echo "  - $file"
  done
  echo ""
  echo "Check $LOG_FILE for detailed error messages"
fi

echo ""
echo "Conversion complete!"
echo "Log file: $LOG_FILE"
echo "========================================================"

# Exit with failure code if any conversions failed
exit $failed
