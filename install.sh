#!/bin/bash
# INSTALL.sh
#   by Tim Müller
# 
# Script for installing the required dependencies in a `venv` environment.
# 


# Always work in the script's directory
cd "${0%/*}"

# Run the clean script
bash ./clean.sh || exit "$?"

# Install the venv
python3 -m venv ./venv || exit "$?"
source venv/bin/activate || exit "$?"
bash -c "SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True python3 -m pip install -r ./requirements.txt" || exit "$?"

# Alrighty that's it
