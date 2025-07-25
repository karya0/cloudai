#!/bin/bash
# generated by CloudAI@__CLOUDAI_VERSION__
#SBATCH --job-name=__JOB_NAME__
#SBATCH --output=__OUTPUT_DIR__/output/stdout.txt
#SBATCH --error=__OUTPUT_DIR__/output/stderr.txt
#SBATCH --partition=main
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH --gres=gpu:8

export SLURM_JOB_MASTER_NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)

srun --export=ALL --mpi=pmix --container-image=nvcr.io/nvidia/pytorch:24.02-py3 --container-mounts=__OUTPUT_DIR__/output:/cloudai_run_results,__OUTPUT_DIR__/install:/cloudai_install,__OUTPUT_DIR__/output --output=__OUTPUT_DIR__/output/mapping-stdout.txt --error=__OUTPUT_DIR__/output/mapping-stderr.txt bash -c "echo \$(date): \$(hostname):node \${SLURM_NODEID}:rank \${SLURM_PROCID}."

srun --export=ALL --mpi=pmix --container-image=nvcr.io/nvidia/pytorch:24.02-py3 --container-mounts=__OUTPUT_DIR__/output:/cloudai_run_results,__OUTPUT_DIR__/install:/cloudai_install,__OUTPUT_DIR__/output --ntasks=1 --ntasks-per-node=1 --output=__OUTPUT_DIR__/output/metadata/node-%N.toml --error=__OUTPUT_DIR__/output/metadata/nodes.err bash /cloudai_install/slurm-metadata.sh

srun --export=ALL --mpi=pmix --container-image=nvcr.io/nvidia/pytorch:24.02-py3 --container-mounts=__OUTPUT_DIR__/output:/cloudai_run_results,__OUTPUT_DIR__/install:/cloudai_install,__OUTPUT_DIR__/output bash -c "source __OUTPUT_DIR__/output/env_vars.sh; /opt/hpcx/ucc/bin/ucc_perftest -c alltoall -b 1 -e 8M -m cuda -F"
