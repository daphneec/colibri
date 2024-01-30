#!/bin/bash
# CLEAN.sh
#   by Tim Müller
# 
# Script for cleaning the repository from installed stuff.
# 


# Always work in the script's directory
cd "${0%/*}"

# Clean the output
echo "[clean.sh] Cleaning 'output' directory"
rm -rf ./output
# Clean the miniconda
miniconda_path="$(pwd)/miniconda_py311"
if [[ -d "$miniconda_path" ]]; then
    echo "[clean.sh] Removing '$miniconda_path' virtual environment"
    rm -rf "$miniconda_path"
else
    echo "[clean.sh] WARNING: Miniconda at '$miniconda_path' not found, not cleaning it"
fi

