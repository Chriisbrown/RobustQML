# Agent Instructions

## Environment Setup

Before running any command in this repository, always execute commands in the `pennylane` conda environment.

Use `conda run` to execute commands in the pennylane environment:

```bash
conda run -n pennylane <command>
```

For Python scripts, use:

```bash
conda run -n pennylane env PYTHONPATH=/scratch/RobustQML python <script.py>
```

For pip installs:

```bash
conda run -n pennylane pip install <package>
```

This ensures all dependencies from the pennylane environment are available.

## HDF5 File Locking

When running scripts that access HDF5 files on EOS (CERN's storage), you may encounter file locking errors. If this happens, set the `HDF5_USE_FILE_LOCKING` environment variable to `FALSE`:

```bash
conda run -n pennylane env HDF5_USE_FILE_LOCKING=FALSE python <script.py>
```

## GPU Support

The code automatically detects if a GPU is available and configures TensorFlow accordingly. On systems with GPUs, training will automatically use the GPU. On CPU-only systems, it falls back gracefully to CPU training.

To use a specific GPU, you can set the `CUDA_VISIBLE_DEVICES` environment variable:

```bash
conda run -n pennylane env CUDA_VISIBLE_DEVICES=0 python train/train.py -y model/configs/VICRegModel.yaml -o output/VICReg
```
