# START.sh
#   by Lut99
#
# Created:
#   05 Dec 2023, 13:04:13
# Last edited:
#   05 Dec 2023, 13:11:52
# Auto updated?
#   Yes
#
# Description:
#   Script for starting the project in either 'normal' or 'secure' mode.
#


# Store the caller's CWD to refer to it when patching their relative paths
old_pwd="$(pwd)"

# Always work in the script's directory
cd "${0%/*}"

# Read the arguments
if [[ "$#" -eq 2 && ("$1" == "normal" || "$1" == "secure") ]]; then
    # Store the two
    mode="$1"
    path="$2"
else
    2>&1 echo "Usage: $0 normal|secure PATH_TO_CONFIG_YAML"
    exit 1
fi

# Resolve the executable
exec="train.py"
if [[ "$mode" == "secure" ]]; then
    exec="secure_train.py"
fi

# Patch the path to allow the user to use relative paths that make sense for them but not in the (possibly different) script directory
if [[ "$path" != /* ]]; then
    path="$old_pwd/$path"
fi

# Use the virtual environment
source ./venv/bin/activate || exit "$?"

# Alright now hit pytorch
torchrun --nnodes 1 --nproc_per_node=1 --node_rank=0 "./$exec" "app:$path"
