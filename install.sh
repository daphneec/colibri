#!/bin/bash
# INSTALL.sh
#   by Tim Müller
# 
# Script for installing the required dependencies in a `venv` environment.
# 


# Read CLIi
distro=
cuda_to_install=latest
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
	elif [[ "$arg" == "-c11" || "$arg" == "--cuda11" ]]; then
	    # Mark that we're using CUDA 11 mode instead
	    cuda_to_install=11
        elif [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
            # Show the help thing
	    echo "Usage: $0 [<OPTS>] [<DISTRO>]"
            echo ""
            echo "Where <DISTRO> is an optional identifier indicating you like to install required system packages:"
            echo " - 'debian' | 'ubuntu' to install with 'apt-get'"
            echo ""
	    echo "Options:"
	    echo "  -c,--clean     Runs the 'clean.sh' script before installing new things."
            echo "  -c11,--cuda11  Installs CUDA toolkit version 11.4 instead of the latest."
	    echo "  -h,--help      Shows this help menu, then exits."
            echo ""
            exit 0
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
done
if [[ "$errored" -eq 1 ]]; then
    exit 1
fi

# Always work in the script's directory
cd "${0%/*}"

# Run the clean script
if [[ "$run_clean" -eq 1 ]]; then
    bash ./clean.sh || exit "$?"
fi

# Install system deps
if [[ "$distro" == "debian" || "$distro" == "ubuntu" ]]; then
    echo "[install.sh] Installing Debian/Ubuntu dependencies with 'apt-get'..."
    sudo apt-get update || exit "$?"
    sudo apt-get install -y python3 python3-pip python3-venv libjpeg-dev zlib1g-dev libgl1-mesa-dev || exit "$?"
elif [[ ! -z "$distro" ]]; then
    2>&1 echo "[install.sh] ERROR: Unknown distro identifier '$distro'"
    exit 1
fi

# Install anaconda with the correct version
miniconda_path="$(pwd)/miniconda_py311"
if [[ ! -d "$miniconda_path" ]]; then
    miniconda_installer_path="$(pwd)/miniconda_py311_installer.sh"
    if [[ ! -f "$miniconda_installer_path" ]]; then
        echo "[install.sh] Downloading miniconda installer to '$miniconda_installer_path'..."
        wget https://repo.anaconda.com/miniconda/Miniconda3-py311_23.11.0-2-Linux-x86_64.sh -O "$miniconda_installer_path" || exit $?
    else
	echo "[install.sh] Miniconda installer at '$miniconda_installer_path' already exists"
    fi

    echo "[install.sh] Installing Miniconda to '$miniconda_path'..."
    bash "$miniconda_installer_path" -b -p "$miniconda_path" || exit $?
else
    echo "[install.sh] Miniconda at '$miniconda_path' already exists, not installing"
fi

# Assert we're now using the correct pip
# export PATH="$miniconda_path/bin:$PATH"
. "$miniconda_path/etc/profile.d/conda.sh"
conda activate
py_path=$(which python3)
echo "[install.sh] Using python3 at '$py_path'"
if [[ "$py_path" != "$miniconda_path/bin/python3" ]]; then
    2>&1 echo "[install.sh] ERROR: Not using python3 from Miniconda; got '$py_path', expected '$miniconda_path/bin/python3'"
    exit 1
fi

echo "[install.sh] Installing torch & torchvision using conda..."
toolkitver=
if [[ "$cuda_to_install" -eq 11 ]]; then toolkitver='=11.8'; fi
conda install -y pytorch torchvision torchaudio "pytorch-cuda$toolkitver" -c pytorch -c nvidia || exit $?

echo "[install.sh] Installing dependencies in requirements.txt..."
bash -c "SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True python3 -m pip install -r ./requirements.txt" || exit "$?"

# Alrighty that's it
echo "[install.sh] Success"

