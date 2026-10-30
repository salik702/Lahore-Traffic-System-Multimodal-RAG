#!/bin/bash
# Collects all evaluation artifacts into a single zip for reporting

EVALUATION_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$EVALUATION_DIR/artifacts"
TIMESTAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
ZIP_FILE="$OUTPUT_DIR/lahore-traffic-evaluation-$TIMESTAMP.zip"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Collect all evaluation files
FILES=(
    "__init__.py"
    "__main__.py"
    "__pycache__.py"
    "commands.py"
    "dataset.py"
    "dataset.json"
    "evaluation_results.json"
    "evaluation_report.md"
    "detailed_evaluation_report.json"
    "metrics.py"
    "main.py"
    "pyproject.toml"
    "requirements.txt"
    "README.md"
    "pipeline_config.yml"
    "CHANGELOG.md"
)

# Create temporary directory
TEMP_DIR="$OUTPUT_DIR/temp_collect"
mkdir -p "$TEMP_DIR"

# Copy all files
for file in "${FILES[@]}"; do
    if [ -f "$EVALUATION_DIR/$file" ]; then
        cp "$EVALUATION_DIR/$file" "$TEMP_DIR/"
    fi
    if [ -d "$EVALUATION_DIR/$file" ]; then
        cp -r "$EVALUATION_DIR/$file" "$TEMP_DIR/"
    fi
done

# Create zip file
cd "$OUTPUT_DIR"
zip -r "$ZIP_FILE" "temp_collect" >/dev/null

# Clean up temporary directory
rm -rf "$TEMP_DIR"

# List created artifacts
echo "Created artifacts in: $OUTPUT_DIR"
echo "Archive file: $ZIP_FILE"

# Show archive contents
unzip -l "$ZIP_FILE" 2>/dev/null | head -20

echo "\nEvaluation artifacts collected successfully!"