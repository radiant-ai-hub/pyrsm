#!/usr/bin/env bash

# Test script to convert basics/*.qmd files to executed .ipynb notebooks
# Outputs to ../basics/ with -qmd.ipynb suffix for easy comparison with existing files

# Configuration
SOURCE_DIR="./basics"
OUTPUT_DIR="../basics"
SUFFIX="-qmd"
LOG_FILE="test_conversion.log"

# Counters
converted=0
failed=0
declare -a failed_files

echo "========================================================"
echo "QMD to IPYNB Conversion Test (Basics Folder)"
echo "========================================================"
echo "Source: $SOURCE_DIR"
echo "Output: $OUTPUT_DIR"
echo "Suffix: $SUFFIX.ipynb"
echo "Log file: $LOG_FILE"
echo "========================================================"
echo ""

# Clear log file
> "$LOG_FILE"

# Find and process all .qmd files in basics folder
while IFS= read -r qmd_file; do
  if [ -z "$qmd_file" ]; then
    continue
  fi

  # Extract directory and filename
  dir=$(dirname "$qmd_file")
  filename=$(basename "$qmd_file" .qmd)

  # Intermediate notebook file (created in same directory as .qmd)
  intermediate_notebook="$dir/${filename}.ipynb"

  # Final output filename with suffix
  output_file="$OUTPUT_DIR/${filename}${SUFFIX}.ipynb"

  echo "Converting: $qmd_file -> $output_file"

  # Step 1: Convert .qmd to .ipynb using quarto (no execution)
  echo "  [Step 1] Converting to .ipynb format..."
  if quarto render "$qmd_file" --to ipynb --no-execute 2>&1 | tee -a "$LOG_FILE"; then

    if [ ! -f "$intermediate_notebook" ]; then
      echo "  ✗ Failed: Notebook file not found at $intermediate_notebook"
      ((failed++))
      failed_files+=("$filename")
      echo ""
      continue
    fi

    # Step 2: Execute the notebook using jupyter nbconvert
    echo "  [Step 2] Executing notebook with jupyter..."
    if jupyter nbconvert --to notebook --execute --inplace "$intermediate_notebook" 2>&1 | tee -a "$LOG_FILE"; then

      # Move the executed notebook to the output directory with -qmd suffix
      mv "$intermediate_notebook" "$output_file"
      echo "  ✓ Success: Converted and executed"
      ((converted++))
    else
      echo "  ✗ Failed during execution: Check $LOG_FILE for details"
      ((failed++))
      failed_files+=("$filename")
      # Clean up any partially created file
      [ -f "$intermediate_notebook" ] && rm "$intermediate_notebook"
    fi
  else
    echo "  ✗ Failed during conversion: Check $LOG_FILE for details"
    ((failed++))
    failed_files+=("$filename")
  fi
  echo ""
done < <(find "$SOURCE_DIR" -maxdepth 1 -name "*.qmd" -type f | sort)

# Summary
echo "========================================================"
echo "Conversion Summary:"
echo "  Successfully converted: $converted"
echo "  Failed: $failed"
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
echo "Next steps:"
echo "  1. Review generated files in $OUTPUT_DIR"
echo "  2. Compare *-qmd.ipynb with original *.ipynb files"
echo "  3. If satisfied, run production script to process all folders"
echo "========================================================"

exit $failed
