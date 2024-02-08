# HR-NAS: Searching Efficient High-Resolution Neural Architectures with Lightweight Transformers (CVPR21 Oral)


## Recent updates
Some recent updates have been done to make it work with [Crypten](https://crypten.ai/). In addition, some files have been changed to allow better compatability with non-GPU machines for local testing and general easier interaction.

### Installation procedure
To run the framework locally, download this repository; e.g.,
```bash
git clone https://github.com/daphneec/colibri && cd colibri
```
Then, you can install the required pip packages using:
```bash
./install.sh
```

Depending on your distribution, you can also automatically install required dependencies. The following platforms are currently supported:
- Debian/Ubuntu: run the script with `./install.sh debian` or `./install.sh ubuntu` to automatically install dependencies using `apt`.

For unsupported platforms, feel free to adapt `install.sh` yourself an make a PR :) Otherwise, install the following dependencies manually:
- [Python](https://python.org) (with `pip` and `venv`) (>= 3.9)
- [libjpeg](https://libjpeg.sourceforge.net/)
- [zlib](https://z-lib.io/)
- [OpenGL](https://www.opengl.org/)

Then you can run the script as above to install the pip packages.

If you are running on an older system, you can use CUDA 11 instead of the latest by adding the `--cuda11`-flag:
```bash
# -c11 works too!
./install.sh --cuda11
```

Finally, you can clean any previous installation by running:
```bash
./install.sh --clean
```

> If you intent to install `requirements.txt` manually, note that `start.sh` expects a miniconda environment to be present. Check `install.sh` for details on how this is installed.


### Running
To run the project, use the `start.sh` script to run it.

It has the following arguments:
```
start.sh [normal|secure|insecure] <PATH_TO_CONFIG_YAML>
```

If you want to run the framework _without_ Crypten, use the `normal` keyword to the `start.sh` script.  
If you want to run _with_ Crypten use the `secure` keyword. 'insecure' uses pytorch layers in the normal hr-nas but evaluates the runtime of layers using the analytical model.

Then, dependening on which dataset you want to use (see below how to install them), select the appropriate configuration file in the `configs` directory. In addition to the ones normally provided by HR-NAS, there are also config files prefixed with `secure_`, which can only be used in the `secure` mode. In addition, files prefixed with `cpu_` are modified to make the framework run in a mode that doesn't require a GPU.

For example, to run a non-Crypten version of a CoCo dataset:
```bash
./start.sh normal configs/keypoint_coco.yml
```
Or, to run a Crypten-modified ImageNet dataset using CPU-only modifications:
```bash
./start.sh secure configs/secure_cpu_cls_imagenet.yml
```

Or, to run a non-Crypten version of a Coco dataset, but using the analytical model:
'''
.start.sh inseucre configs/keypoint_coco.yml
'''

### Running - Cluster
If you're running the project on a SLURM-cluster, you can make your life easier by using the provided `runit.sh` and `launch.sh` scripts.

`launch.sh` wraps the `start.sh` and makes it a distributed SLURM job script, for use with `sbatch`. You can change this file to change properties of the resource requirements (see the `# SBATCH=...` comments).

`runit.sh` in turn wraps `launch.sh`, to launch it using `sbatch` and then attach to the output log file.

Note, however, that to be directory-agnostic, `launch.sh` needs to be preprocessed with the current folder location. To do so without `runit.sh`, manually run:
```bash
sed -i "s+%CURRENT_FOLDER%+$(pwd)+g" <PATH_TO_LAUNCH.SH FILE>
```
where `<PATH_TO_LAUNCH.SH FILE>` should be replaced with the path in question.


### Modifications
This version of the framework shows the following modifications over those of the forked repository:
1. _Crypten-specific layers_
    1. Added lots of Crypten-specific counterparts (all Python files starting with `secure_`).
        - For specific changes, use `diff --color -u <file>.py secure_<file>.py`
    2. Custom `LayerNorm` layer to implement Crypten-compatible version (`models/secure_layernorm.py`).
        - Implemented in terms of `BatchNorm` using transposes.
    3. Custom `MultiHeadAttention` layer to implement Crypten-compatible version (`models/secure_multi_head_attention.py`).
        - Taken from MPC Former and adapted for Crypten and this specific use-case.
    4. Custom `ZeroPad2d` layer to implement Crypten-compatible version (`models/secure_padding.py`).
    - Simply thin wrapper around `crypten.nn.ConstantPad`.
    5. Custom `UpsampleNearest` layer to implement Crypten-compatible `Upsample(type="nearest")` layer (`models/secure_upsample.py`).
        - Implemented using custom algorithm to emulate same behaviour.
2. _Other fixes_
    1. Added `secure_`, `cpu_` and `secure_cpu_` files to the `configs/` directory.
    2. Changed parts of the framework to work with CPU-only mode (`common.py`, `train.py`, `val.py`, `validation.py`, `mmseg/validation.py`, `utils/dataflow.py`, `utils/distributed.py`, `utils/model_profiling.py`, and any `secure_` version of these files).
    3. Fixed broken dependencies in `requirements.txt`.
    4. Fixed broken usage of updated pickle package (`utils/lmdb_dataset.py`).
    5. Added option to generate fake `cls_imagenet.yml`-compatible dataset with `tools/generate_fake_data.py` (note: change the dataset type to the non-lmdb version!).
3. _Convenience additions_
    1. Added scripts for `launch.sh`ing, `install.sh`ing, `start.sh`ing and `clean.sh`ing.

Plus other stuff I have not mention here, but is probably small and insignificant. See diffs with `secure_` version with non-`secure_` version for specific changes.

## Environment
Require Python3, CUDA>=10.1, and torch>=1.4, all dependencies are as follows:
```shell script
pip3 install torch==1.4.0 torchvision==0.5.0 opencv-python tqdm tensorboard lmdb pyyaml packaging Pillow==6.2.2 matplotlib yacs pyarrow==0.17.1
pip3 install cityscapesscripts  # for Cityscapes segmentation
pip3 install mmcv-full==latest+torch1.4.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html  # for Segmentation data loader
pip3 install pycocotools shapely==1.6.4 Cython pandas pyyaml json_tricks scikit-image  # for COCO keypoint estimation
```
or ```pip3 install requirements.txt```

## Setup

Optionally configure NCCL before running:
```shell script
export NCCL_IB_DISABLE=1 
export NCCL_IB_HCA=mlx5_0 
export NCCL_IB_GID_INDEX=3 
export NCCL_SOCKET_IFNAME=eth0 
export HOROVOD_MPI_THREADS_DISABLE=1
export OMP_NUM_THREADS=56
export KMP_AFFINITY=granularity=fine,compact,1,0
```
Set the following ENV variable:
```
$MASTER_ADDR: IP address of the node 0 (Not required if you have only one node (machine))
$MASTER_PORT: Port used for initializing distributed environment
$NODE_RANK: Index of the node
$N_NODES: Number of nodes 
$NPROC_PER_NODE: Number of GPUs (NOTE: should exactly match local GPU numbers with `CUDA_VISIBLE_DEVICES`)
```

Example1 (One machine with 8 GPUs):
```
Node 1: 
>>> python -m torch.distributed.launch --nproc_per_node=8
--nnodes=1 --node_rank=0 --master_port=1234 train.py
```

Example2 (Two machines, each has 8 GPUs):
```
Node 1: (IP: 192.168.1.1, and has a free port: 1234)
>>> python -m torch.distributed.launch --nproc_per_node=8
--nnodes=2 --node_rank=0 --master_addr="192.168.1.1"
--master_port=1234 train.py

Node 2:
>>> python -m torch.distributed.launch --nproc_per_node=8
--nnodes=2 --node_rank=1 --master_addr="192.168.1.1"
--master_port=1234 train.py
```

## Datasets

0. Fake dataset (ImageNet)
    - Generate fake data in torchvision ImageFolder format by running `tools/generate_fake_data.py ./data/fake -f` in the repo root
    - Convert to lmdb by running `utils/lmdb_dataset.py --src_dir ./data/fake --dst_dir ./data/imagenet_lmdb` in the repo root
    - Now run the YAMLs with `dataset:imagenet1k_lmdb`

1. ImageNet
    - Prepare ImageNet data following pytorch [example](https://github.com/pytorch/examples/tree/master/imagenet).
    - Optional: Generate lmdb dataset by `utils/lmdb_dataset.py`. If not, please overwrite `dataset:imagenet1k_lmdb` in yaml to `dataset:imagenet1k`.
    - The directory structure of `$DATA_ROOT` should look like this:
    ```
    ${DATA_ROOT}
    ├── imagenet
    └── imagenet_lmdb
    ```
    - Link the data:
    ```shell script
    ln -s YOUR_LMDB_DIR data/imagenet_lmdb
    ```
   
2. Cityscapes
    - Download data from [Cityscapes](https://www.cityscapes-dataset.com/).
    - unzip gtFine_trainvaltest.zip leftImg8bit_trainvaltest.zip
    - Link the data:
    ```shell script
    ln -s YOUR_DATA_DIR data/cityscapes
    ```
    - preprocess the data:
    ```shell script
    python3 tools/convert_cityscapes.py data/cityscapes --nproc 8
    ```
      
3. ADE20K
    - Download data from [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/).
    - unzip ADEChallengeData2016.zip
    - Link the data:
    ```shell script
    ln -s YOUR_DATA_DIR data/ade20k
    ```

4. COCO keypoints
    - Download data from [COCO](https://cocodataset.org/#download).
    - build tools
    ```shell script
    git clone https://github.com/cocodataset/cocoapi.git
    cd cocoapi/PythonAPI
    python3 setup.py build_ext --inplace
    python3 setup.py build_ext install
    make  # for nms
    ```
    - Unzip and Link the data:
    ```shell script
    ln -s YOUR_DATA_DIR data/coco
    ```
    - We also provide person detection result of COCO val2017 and test-dev2017 to reproduce our multi-person pose estimation results. Please download from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzzDXoz5BeFl8sWM-) or [GoogleDrive](https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk?usp=sharing).
    - Download and extract them under ```data/coco/person_detection_results```, and make them look like this:
    ```
    ${POSE_ROOT}
    |-- data
    `-- |-- coco
        `-- |-- annotations
            |   |-- person_keypoints_train2017.json
            |   `-- person_keypoints_val2017.json
            |-- person_detection_results
            |   |-- COCO_val2017_detections_AP_H_56_person.json
            |   |-- COCO_test-dev2017_detections_AP_H_609_person.json
            `-- images
                |-- train2017
                |   |-- 000000000009.jpg
                |   |-- 000000000025.jpg
                |   |-- 000000000030.jpg
                |   |-- ... 
                `-- val2017
                    |-- 000000000139.jpg
                    |-- 000000000285.jpg
                    |-- 000000000632.jpg
                    |-- ... 
    ```

## Running (train & evaluation)
- Search for NAS models.
    ```shell script
    python3 -m torch.distributed.launch --nproc_per_node=${NPROC_PER_NODE} --nnodes=${N_NODES} \
        --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
        --use_env train.py app:configs/YOUR_TASK.yml
    ```
    Supported tasks:
    - cls_imagenet
    - seg_cityscapes
    - seg_ade20k
    - keypoint_coco
    
    The super network is constructed using ```model_kwparams``` in YOUR_TASK.yml.  
    To enable the searching of Transformer, set ```prune_params.use_transformer=True``` in YOUR_TASK.yml,
  the token numbers of each Transformer will be printed during training.  
    The searched architecture can be found in ```best_model.json``` in the output dir.  
  

- Retrain the searched models.
    - For retraining the searched classification model, please use ```best_model.json``` to overwrite the ```checkpoint.json``` in root dir of this project.
    
    - Modify ```models/hrnet.py``` to set ```checkpoint_kwparams = json.load(open('checkpoint.json'))``` and 
    ```class InvertedResidual(InvertedResidualChannelsFused)```
    
    - Retrain the model.
    ```shell script
    python3 -m torch.distributed.launch --nproc_per_node=${NPROC_PER_NODE} --nnodes=${N_NODES} \
        --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
        --use_env train.py app:configs/cls_retrain.yml
    ```

## Miscellaneous
1. Plot keypoint detection results.
    ```shell script
    python3 tools/plot_coco.py --prediction output/results/keypoints_val2017_results_0.json --save-path output/vis/
    ```

2. About YAML config
- The codebase is a general ImageNet training framework using yaml config with several extension under `apps` dir, based on PyTorch.
    - YAML config with additional features
        - `${ENV}` in yaml config.
        - `_include` for hierachy config.
        - `_default` key for overwriting.
        - `xxx.yyy.zzz` for partial overwriting.
    - `--{{opt}} {{new_val}}` for command line overwriting.

3. Any questions regarding HR-NAS, feel free to contact the author (mingyuding@hku.hk).
4. If you find our work useful in your research please consider citing our paper:
    ```
    @inproceedings{ding2021hrnas,
      title={HR-NAS: Searching Efficient High-Resolution Neural Architectures with Lightweight Transformers},
      author={Ding, Mingyu and Lian, Xiaochen and Yang, Linjie and Wang, Peng and Jin, Xiaojie and Lu, Zhiwu and Luo, Ping},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2021}
    }
    ```
