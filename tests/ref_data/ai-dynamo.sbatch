#!/bin/bash
# generated by CloudAI@__CLOUDAI_VERSION__
#SBATCH --job-name=__JOB_NAME__
#SBATCH --output=__OUTPUT_DIR__/output/stdout.txt
#SBATCH --error=__OUTPUT_DIR__/output/stderr.txt
#SBATCH --partition=main
#SBATCH -N 2
#SBATCH --gpus-per-node=8
#SBATCH --gres=gpu:8

export SLURM_JOB_MASTER_NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)

srun --export=ALL --mpi=pmix --container-image=nvcr.io/nvidia/ai-dynamo:24.09 --container-mounts=__OUTPUT_DIR__/output:/cloudai_run_results,__OUTPUT_DIR__/install:/cloudai_install,__OUTPUT_DIR__/output,__OUTPUT_DIR__/output/hf_home:/root/.cache/huggingface,__OUTPUT_DIR__/output/run.sh:/opt/run.sh,__OUTPUT_DIR__/output/dynamo_config.yaml:__OUTPUT_DIR__/output/dynamo_config.yaml --output=__OUTPUT_DIR__/output/mapping-stdout.txt --error=__OUTPUT_DIR__/output/mapping-stderr.txt bash -c "echo \$(date): \$(hostname):node \${SLURM_NODEID}:rank \${SLURM_PROCID}."

srun --export=ALL --mpi=pmix --container-image=nvcr.io/nvidia/ai-dynamo:24.09 --container-mounts=__OUTPUT_DIR__/output:/cloudai_run_results,__OUTPUT_DIR__/install:/cloudai_install,__OUTPUT_DIR__/output,__OUTPUT_DIR__/output/hf_home:/root/.cache/huggingface,__OUTPUT_DIR__/output/run.sh:/opt/run.sh,__OUTPUT_DIR__/output/dynamo_config.yaml:__OUTPUT_DIR__/output/dynamo_config.yaml --ntasks=2 --ntasks-per-node=1 --output=__OUTPUT_DIR__/output/metadata/node-%N.toml --error=__OUTPUT_DIR__/output/metadata/nodes.err bash /cloudai_install/slurm-metadata.sh

srun --export=ALL --mpi=pmix --container-image=nvcr.io/nvidia/ai-dynamo:24.09 --container-mounts=__OUTPUT_DIR__/output:/cloudai_run_results,__OUTPUT_DIR__/install:/cloudai_install,__OUTPUT_DIR__/output,__OUTPUT_DIR__/output/hf_home:/root/.cache/huggingface,__OUTPUT_DIR__/output/run.sh:/opt/run.sh,__OUTPUT_DIR__/output/dynamo_config.yaml:__OUTPUT_DIR__/output/dynamo_config.yaml --nodes=2 --ntasks=2 --ntasks-per-node=1 /opt/run.sh 