#!/bin/bash
# INSTALL.sh
#   by Tim Müller
# 
# Script for installing the required dependencies in a `venv` environment.
# 


# Read CLIi
distro=
run_clean=0
allow_opts=1
pos_i=0
errored=0
for arg in $@; do
    # Match on option or not
    if [[ "$allow_opts" -eq 1 && "$arg" =~ ^- ]]; then
	if [[ "$arg" == "-c" || "$arg" == "--clean" ]]; then
	    # Mark we seen it
	    run_clean=1
	elif [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
            # Show the help thing
	    echo "Usage: $0 [<OPTS>] [<DISTRO>]"
            echo ""
            echo "Where <DISTRO> is an optional identifier indicating you like to install required system packages:"
            echo " - 'debian' | 'ubuntu' to install with 'apt-get'"
            echo ""
	    echo "Options:"
	    echo "  -c,--clean: Runs the 'clean.sh' script before installing new things."
	    echo ""
            exit 1
	elif [[ "$arg" == "--" ]]; then
	    # No longer allow the options
	    allow_opts=0
	else
	    >&2 echo "Unknown option '$arg'"
	    errored=1
	fi
    else
	# Recognize the positional argument
	if [[ "$pos_i" -eq 0 ]]; then
	    # It's the distro
	    distro="$arg"
	else
	    >&2 echo "Unexpected positional argument '$arg'"
	    errored=1
	fi
	pos_i=$((pos_i+1))
    fi
    echo "$arg"
done
if [[ "$errored" -eq 1 ]]; then
    exit 1
fi

# Always work in the script's directory
cd "${0%/*}"

# Run the clean script
if [[ "$clean" -eq 1 ]]; then
    bash ./clean.sh || exit "$?"
fi

# Install system deps
if [[ "$distro" == "debian" || "$distro" == "ubuntu" ]]; then
    sudo apt-get update || exit "$?"
    sudo apt-get install -y python3 python3-pip python3-venv libjpeg-dev zlib1g-dev libgl1-mesa-dev || exit "$?"
elif [[ ! -z "$distro" ]]; then
    echo "Unknown distro identifier '$distro'"
    exit 1
fi

# Install the venv
python3 -m venv ./venv || exit "$?"
source venv/bin/activate || exit "$?"
bash -c "SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True python3 -m pip install -r ./requirements.txt" || exit "$?"

# Alrighty that's it
