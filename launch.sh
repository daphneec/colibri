#!/bin/bash

#SBATCH -N 1
#SBATCH --gres=gpu:pascal:4
#SBATCH --cpus-per-task=4
#SBATCH -t 150:00:00
#SBATCH --mem=240G
#SBATCH -p all
#SBATCH -o /home/tmuller/colibri/logs_%j.out

# GLOBALS
NUM_NODES=1
GPUS_PER_NODE=4
MODE=normal                            # Change to 'secure' for CrypTen
CONFIG_FILE=configs/cls_imagenet.yml   # Change to 'secure_....yml' for CrypTen

# Go to the folder
cd /home/tmuller/colibri

# Extract the properties we need
MASTER_ADDR=$(python3 - "$SLURM_NODELIST" << EOF
# From: https://gist.github.com/kiyoon/add4a18a5224728fb5d9d9296d0b8a54

import argparse
def get_parser():
    parser = argparse.ArgumentParser(description="Extract master node name from Slurm node list",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("nodelist", help="Slurm nodelist")
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    first_nodelist = args.nodelist.split(',')[0]

    if '[' in first_nodelist:
        a = first_nodelist.split('[')
        first_node = a[0] + a[1].split('-')[0]

    else:
        first_node = first_nodelist

    print(first_node)
EOF
)
echo "[launch.sh] Extracted master address '$MASTER_ADDR'"
MASTER_PORT=1234
echo "[launch.sh] Settled on master port '$MASTER_PORT'"
RANK="$SLURM_NODEID"
echo "[launch.sh] Extracted node rank '$RANK'"

# Run start.sh
cmd="./start.sh \"$MODE\" \"$CONFIG_FILE\" -N \"$NUM_NODES\" --gpus \"$GPUS_PER_NODE\" --rank \"$RANK\" --master-addr \"$MASTER_ADDR\" --master-port \"$MASTER_PORT\""
echo "[launch.sh] Continuing as '$cmd'"
srun bash -c "$cmd" || exit $?

