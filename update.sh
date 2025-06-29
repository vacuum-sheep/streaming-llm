#!/bin/bash

# Update script for streaming-llm versions
# Usage: ./update.sh [o3|r1|cursor] [-r]
#   -r: Reverse copy (copy from version files to base files)

if [ $# -eq 0 ]; then
    echo "Usage: $0 [o3|r1|cursor] [-r]"
    echo "  o3: Copy base files to o3 versions"
    echo "  r1: Copy base files to r1 versions"
    echo "  cursor: Copy base files to cursor versions"
    echo "  -r: Reverse copy (copy from version files to base files)"
    exit 1
fi

VERSION=$1
REVERSE=false

# Check for reverse flag
if [ "$1" = "-r" ]; then
    REVERSE=true
    VERSION=$2
    if [ -z "$VERSION" ]; then
        echo "Error: Version must be specified when using -r flag"
        echo "Usage: $0 -r [o3|r1|cursor]"
        exit 1
    fi
fi

case $VERSION in
    "o3")
        if [ "$REVERSE" = true ]; then
            echo "Reverse updating o3 version..."
            # Copy o3 files to base files
            cp streaming_llm/pos_shift/modify_qwen_o3.py streaming_llm/pos_shift/modify_qwen.py
            cp streaming_llm/utils_o3.py streaming_llm/utils.py
            cp streaming_llm/kv_cache_o3.py streaming_llm/kv_cache.py
            echo "Base files updated from o3 version successfully!"
        else
            echo "Updating o3 version..."
            # Copy base files to o3 versions
            cp streaming_llm/pos_shift/modify_qwen.py streaming_llm/pos_shift/modify_qwen_o3.py
            cp streaming_llm/utils.py streaming_llm/utils_o3.py
            cp streaming_llm/kv_cache.py streaming_llm/kv_cache_o3.py
            echo "o3 version updated successfully!"
        fi
        ;;
    "r1")
        if [ "$REVERSE" = true ]; then
            echo "Reverse updating r1 version..."
            # Copy r1 files to base files
            cp streaming_llm/pos_shift/modify_qwen_r1.py streaming_llm/pos_shift/modify_qwen.py
            cp streaming_llm/utils_r1.py streaming_llm/utils.py
            cp streaming_llm/kv_cache_r1.py streaming_llm/kv_cache.py
            echo "Base files updated from r1 version successfully!"
        else
            echo "Updating r1 version..."
            # Copy base files to r1 versions
            cp streaming_llm/pos_shift/modify_qwen.py streaming_llm/pos_shift/modify_qwen_r1.py
            cp streaming_llm/utils.py streaming_llm/utils_r1.py
            cp streaming_llm/kv_cache.py streaming_llm/kv_cache_r1.py
            echo "r1 version updated successfully!"
        fi
        ;;
    "cursor")
        if [ "$REVERSE" = true ]; then
            echo "Reverse updating cursor version..."
            # Copy cursor files to base files
            cp streaming_llm/pos_shift/modify_qwen_cursor.py streaming_llm/pos_shift/modify_qwen.py
            cp streaming_llm/utils_cursor.py streaming_llm/utils.py
            cp streaming_llm/kv_cache_cursor.py streaming_llm/kv_cache.py
            echo "Base files updated from cursor version successfully!"
        else
            echo "Updating cursor version..."
            # Copy base files to cursor versions
            cp streaming_llm/pos_shift/modify_qwen.py streaming_llm/pos_shift/modify_qwen_cursor.py
            cp streaming_llm/utils.py streaming_llm/utils_cursor.py
            cp streaming_llm/kv_cache.py streaming_llm/kv_cache_cursor.py
            echo "cursor version updated successfully!"
        fi
        ;;
    *)
        echo "Invalid version: $VERSION"
        echo "Valid versions: o3, r1, cursor"
        exit 1
        ;;
esac 
