#!/bin/bash

# Switch script for streaming-llm versions
# Usage: ./switch.sh [source_version] [target_version]
# Example: ./switch.sh r1 o3
#   This will: bash update.sh r1 (copy base to r1), then bash update.sh -r o3 (copy o3 to base)

if [ $# -ne 2 ]; then
    echo "Usage: $0 [source_version] [target_version]"
    echo "  source_version: o3, r1, or cursor"
    echo "  target_version: o3, r1, or cursor"
    echo ""
    echo "Example: $0 r1 o3"
    echo "  This will copy from r1 version to o3 version"
    exit 1
fi

SOURCE_VERSION=$1
TARGET_VERSION=$2

# Validate versions
VALID_VERSIONS=("o3" "r1" "cursor")
SOURCE_VALID=false
TARGET_VALID=false

for version in "${VALID_VERSIONS[@]}"; do
    if [ "$SOURCE_VERSION" = "$version" ]; then
        SOURCE_VALID=true
    fi
    if [ "$TARGET_VERSION" = "$version" ]; then
        TARGET_VALID=true
    fi
done

if [ "$SOURCE_VALID" = false ]; then
    echo "Error: Invalid source version '$SOURCE_VERSION'"
    echo "Valid versions: o3, r1, cursor"
    exit 1
fi

if [ "$TARGET_VALID" = false ]; then
    echo "Error: Invalid target version '$TARGET_VERSION'"
    echo "Valid versions: o3, r1, cursor"
    exit 1
fi

echo "Switching from $SOURCE_VERSION to $TARGET_VERSION..."
bash update.sh "$SOURCE_VERSION"
bash update.sh -r "$TARGET_VERSION"
echo "Switch completed successfully!"
