#!/bin/bash

#SBATCH --job-name=qwen_sft_8b_std
#SBATCH --mail-type=ALL
#SBATCH --mail-user=${USER_EMAIL:-}
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=48
#SBATCH --mem=1000gb
#SBATCH --time=48:00:00
#SBATCH --output=/project/log/alignment/qwen_sft_8b_std_%j.out
#SBATCH --partition=batch

HOME=/project
SCRIPTPATH=${HOME}/experiments

echo $SCRIPTPATH
echo $SLURM_JOB_NUM_NODES
echo $SLURM_NTASKS_PER_NODE
echo $SLURM_GPUS_ON_NODE
source ${SCRIPTPATH}/util.sh

CONTAINER=${HOME}/container/ouro_container

export TRITON_HOME=${HOME}/cache/triton
export TRITON_CACHE_DIR=${HOME}/cache/triton/cache
export TRITON_DUMP_DIR=${HOME}/cache/triton/dump
export HF_DATASETS_CACHE=${HOME}/cache/dataset
export HF_HOME=${HOME}/cache/huggingface

srun --jobid $SLURM_JOB_ID apptainer exec -B $SLURM_TMPDIR:/cache --nv $CONTAINER \
    bash ${SCRIPTPATH}/loop_transformer/qwen/slurm/run_sft_8b_std.sh
