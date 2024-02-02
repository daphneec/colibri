# START.sh
#   by Lut99
#
# Created:
#   05 Dec 2023, 13:04:13
# Last edited:
#   12 Dec 2023, 10:24:25
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



### CLI ARGUMENT PARSING ###
mode=
path=
N=1
master_addr="127.0.0.1"
master_port=1234
rank=0
gpus=1
state=start
parsing_arg=
allow_opts=1
pos_i=0
errored=0
i=0
args=( $@ )
while [[ "$i" -lt "${#args[@]}" ]]; do
    arg="${args[$i]}"

    # Match on the parse state first
    if [[ "$state" == "start" ]]; then
        if [[ "$allow_opts" -eq 1 && "$arg" =~ ^- ]]; then
            # It's an option
            if [[ "$arg" == "-N" || "$arg" == "--nnodes" ]]; then
	        # Parse the argument in the next iter
                parsing_arg="nnodes"
                state="arg"
            elif [[ "$arg" == "-a" || "$arg" == "--master-addr" ]]; then
                # Parse the argument in the next iter
                parsing_arg="master-addr"
                state="arg"
            elif [[ "$arg" == "-p" || "$arg" == "--master-port" ]]; then
                # Parse the argument in the next iter
                parsing_arg="master-port"
                state="arg"
            elif [[ "$arg" == "-r" || "$arg" == "--rank" ]]; then
                # Parse the argument in the next iter
                parsing_arg="rank"
                state="arg"
            elif [[ "$arg" == "-g" || "$arg" == "--gpus" ]]; then
                # Parse the argument in the next iter
                parsing_arg="gpus"
                state="arg"
            elif [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
	        echo "Usage: $0 [OPTS] <normal|secure> <CONFIG_PATH>"
                echo ""
                echo "Arguments:"
                echo "  - <normal|secure>  Which mode to enable. Use 'normal' to run the (almost) vanilla HR-NAS,"
                echo "                     or 'secure' to run the CrypTen version."
                echo "  - <CONFIG_PATH>    The path to the config file denoting which model to use. See 'README.md'"
                echo "                     for more information."
                echo ""
                echo "Options:"
                echo "  -N,--nnodes <N>  The number of nodes that are part of this instance. Default: '1'"
                echo "  -a,--master-addr <ADDR>"
                echo "                   The address of the master node for this session. Default: '127.0.0.1'"
                echo "  -p,--master-port <PORT>"
                echo "                   The port number on which to reach the master node for this session. Default: '1234'"
                echo "  -r,--rank <NUM>  The rank of this node. Should be a number, starting at 0 for the first node. Default: '0'"
                echo "  -g,--gpus <N>    Sets the number of GPUs to use -per node-. Default: '1'"
                echo "  -h,--help        Shows this help menu, then exits."
                echo ""
                exit 0
            else
	        2>&1 echo "Unknown option '$arg'"
                errored=1
            fi
        else
	    # Parse the positional index
	    if [[ "$pos_i" -eq 0 ]]; then
                mode="$arg"
	    elif [[ "$pos_i" -eq 1 ]]; then
	        path="$arg"
	    else
	        2>&1 echo "Unknown positional '$arg' at position $pos_i"
                errored=1
	    fi
	    pos_i=$((pos_i+1))
        fi

        # Don't forget to increment i
        i=$((i+1))
    elif [[ "$state" == "arg" ]]; then
        # See if it's a '--'
        if [[ "$allow_opts" -eq 1 && "$arg" == "--" ]]; then
            allow_opts=0
            i=$((i+1))
        elif [[ "$arg" =~ ^- ]]; then
	    # Not given!
            2>&1 echo "Missing value for '--$parsing_arg'"
            errored=1
            # Dont increment i tho, to parse the option correctly
            state="start"
        else
            # Match on the argument
            if [[ "$parsing_arg" == "nnodes" ]]; then
                N="$arg"
            elif [[ "$parsing_arg" == "master-addr" ]]; then
                master_addr="$arg"
            elif [[ "$parsing_arg" == "master-port" ]]; then
                master_port="$arg"
            elif [[ "$parsing_arg" == "rank" ]]; then
                rank="$arg"
            elif [[ "$parsing_arg" == "gpus" ]]; then
                gpus="$arg"
            else
                2>&1 echo "[start.sh] PANIC: Unknown parsing argument '$parsing_arg'"
                exit 1
            fi

            # Increment i before contuining
            i=$((i+1))
            state="start"
        fi
    else
        2>&1 echo "[start.sh] PANIC: Unknown parse state '$state'"
        exit 1
    fi
done
if [[ -z "$mode" ]]; then
    2>&1 echo "Missing mandatory positional argument '<normal|secure>'"
    errored=1
fi
if [[ -z "$path" ]]; then
    2>&1 echo "Missing mandatory positional argument '<CONFIG_PATH>'"
    errored=1
fi
if [[ "$errored" -eq 1 ]]; then
    exit 1
fi





### EXECUTION ###
# Resolve the master address to an IP, always
if [[ ! "$master_addr" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    ip="$(host -4 $master_addr | awk '/has address/ { print $4 }')"
    echo "[start.sh] Resolved '$master_addr' as '$ip'"
    master_addr="$ip"
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

# Find anaconda
echo "[start.sh] Initializing miniconda..."
miniconda_path="$(pwd)/miniconda_py311"
if [[ ! -d "$miniconda_path" ]]; then
    2>&1 echo "[start.sh] ERROR: Miniconda environment '$miniconda_path' not found"
    exit 1
fi
. "$miniconda_path/etc/profile.d/conda.sh"
conda activate || exit "$?"
py_path="$(which python3)"
echo "[start.sh] Using python3 from '$py_path'"
if [[ "$py_path" != "$miniconda_path/bin/python3" ]]; then
    2>&1 echo "[start.sh] PANIC: Not using python3 from Miniconda; got '$py_path', expected '$miniconda_path/bin/python3'"
    exit 1
fi

# Alright now hit pytorch
if [[ "$path" =~ "cpu" ]]; then
    export DEVICE_MODE="cpu"
fi
cmd="torchrun --nnodes=$N --nproc_per_node=$gpus --node_rank=$rank --master-addr=$master_addr --master-port=$master_port \"./$exec\" \"app:$path\""
echo "[start.sh] Launching '$cmd'"
bash -c "$cmd" || exit "$?"

