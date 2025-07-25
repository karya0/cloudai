#!/bin/bash
# generated by CloudAI@__CLOUDAI_VERSION__
#SBATCH --job-name=__JOB_NAME__
#SBATCH --output=__OUTPUT_DIR__/output/stdout.txt
#SBATCH --error=__OUTPUT_DIR__/output/stderr.txt
#SBATCH --partition=main
#SBATCH -N 3
#SBATCH --gpus-per-node=8
#SBATCH --gres=gpu:8

export SLURM_JOB_MASTER_NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export HEAD_NODE=$SLURM_JOB_MASTER_NODE
export NIM_LEADER_IP_ADDRESS=$SLURM_JOB_MASTER_NODE
export NIM_NUM_COMPUTE_NODES=2
export NIM_MODEL_TOKENIZER='deepseek-ai/DeepSeek-R1'
export NIM_CACHE_PATH=__OUTPUT_DIR__/output
export NIM_MODEL_NAME=__OUTPUT_DIR__/output
srun --export=ALL --mpi=pmix --output=__OUTPUT_DIR__/output/mapping-stdout.txt --error=__OUTPUT_DIR__/output/mapping-stderr.txt bash -c "echo \$(date): \$(hostname):node \${SLURM_NODEID}:rank \${SLURM_PROCID}."

srun --export=ALL --mpi=pmix --ntasks=3 --ntasks-per-node=1 --output=__OUTPUT_DIR__/output/metadata/node-%N.toml --error=__OUTPUT_DIR__/output/metadata/nodes.err bash __OUTPUT_DIR__/install/slurm-metadata.sh

srun --export=ALL --mpi=pmix --container-image=nvcr.io/nim/deepseek-ai/deepseek-r1:1.7.2 --container-mounts=__OUTPUT_DIR__/output:/cloudai_run_results,__OUTPUT_DIR__/install:/cloudai_install,__OUTPUT_DIR__/output,__OUTPUT_DIR__/output:__OUTPUT_DIR__/output:ro,__OUTPUT_DIR__/output:__OUTPUT_DIR__/output:rw,__OUTPUT_DIR__/output/start_server_wrapper.sh:/opt/nim/start_server_wrapper.sh:ro --nodes=2 --ntasks=2 --ntasks-per-node=1 /opt/nim/start_server_wrapper.sh &

sleep 3300

srun --export=ALL --mpi=pmix --container-image=nvcr.io/nvidia/tritonserver:25.01-py3-sdk --container-mounts=__OUTPUT_DIR__/output:/cloudai_run_results,__OUTPUT_DIR__/install:/cloudai_install,__OUTPUT_DIR__/output,__OUTPUT_DIR__/output:__OUTPUT_DIR__/output:ro,__OUTPUT_DIR__/output:__OUTPUT_DIR__/output:rw,__OUTPUT_DIR__/output/start_server_wrapper.sh:/opt/nim/start_server_wrapper.sh:ro --nodes=1 --ntasks=1 genai-perf profile -m model --endpoint-type chat --service-kind openai --streaming -u $SLURM_JOB_MASTER_NODE:8000 --num-prompts 20 --synthetic-input-tokens-mean 128 --synthetic-input-tokens-stddev 0 --concurrency 1 --output-tokens-mean 128 --extra-inputs max_tokens:128 --extra-inputs min_tokens:128 --extra-inputs ignore_eos:true --artifact-dir /cloudai_run_results --tokenizer tok -- -v --max-threads 1 --request-count 20
