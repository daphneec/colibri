#!/bin/bash
# CLEAN.sh
#   by Tim Müller
# 
# Script for cleaning the repository from installed stuff.
# 


# Always work in the script's directory
cd "${0%/*}"

# Clean the output
output_path="$(pwd)/output"
echo "[clean.sh] Cleaning '$output_path' directory"
rm -rf "$output_path"

# Clean the miniconda
miniconda_path="$(pwd)/miniconda_py311"
if [[ -d "$miniconda_path" ]]; then
    echo "[clean.sh] Removing '$miniconda_path' virtual environment"
    rm -rf "$miniconda_path"
else
    echo "[clean.sh] WARNING: Miniconda at '$miniconda_path' not found, not cleaning it"
fi

# Clean the local install file
install_path="$(pwd)/install.sh.local"
echo "[clean.sh] Cleaning local install script '$install_path'..."
rm -f "$install_path"

