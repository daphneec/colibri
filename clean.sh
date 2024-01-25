#!/bin/bash
# CLEAN.sh
#   by Tim Müller
# 
# Script for cleaning the repository from installed stuff.
# 


# Always work in the script's directory
cd "${0%/*}"

# Clean the output
echo "Cleaning 'output' directory"
rm -rf ./output
# Clean the venv
echo "Cleaning 'venv' directory"
rm -rf ./venv
