#!/usr/bin/env bash

# Script to convert all .ipynb files to Quarto format (.qmd)
# This script finds all Jupyter notebooks in the current directory and subfolders
# and converts them to Quarto markdown files

# Counter for tracking conversions
converted=0
failed=0

echo "Starting conversion of .ipynb files to Quarto format..."
echo "========================================================"

# Find all .ipynb files and convert them
while IFS= read -r notebook; do
  if [ -z "$notebook" ]; then
    continue
  fi

  # Extract directory and filename
  dir=$(dirname "$notebook")
  filename=$(basename "$notebook" .ipynb)

  # Full output path
  output_file="$dir/$filename.qmd"

  echo "Converting: $notebook -> $output_file"

  if quarto convert "$notebook" -o "$output_file" > /dev/null 2>&1; then
    echo "  ✓ Success"
    ((converted++))
  else
    echo "  ✗ Failed"
    ((failed++))
  fi
  echo
done < <(find . -name "*.ipynb" -type f | sort)

# Summary
echo "========================================================"
echo "Conversion Summary:"
echo "  Successfully converted: $converted"
echo "  Failed: $failed"
echo "========================================================"

if [ $failed -eq 0 ]; then
  echo "All conversions completed successfully!"
  exit 0
else
  echo "Some conversions failed. Please check the output above."
  exit 1
fi
