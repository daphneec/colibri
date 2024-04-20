#!/bin/bash
# source /home/eloise/colibri-eloise/common.sh
echo $MASTER_ADDR $MASTER_PORT $WORLD_SIZE
# DEVICE_MODE=cpu PYTHONWARNINGS=ignore::Warning torchrun --nnodes=1 --nproc_per_node=1 --node_rank=0 --master-port=1234 secure_test_pretrained.py app:configs/secure_cpu_seg_cityscapes.yml
DEVICE_MODE=cpu python3 secure_test_pretrained.py app:configs/secure_cpu_seg_cityscapes.yml
# python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 \
#     --node_rank=0 \
#     --use_env secure_test_pretrained.py app:configs/secure_cpu_seg_cityscapes.yml