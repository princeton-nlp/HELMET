#!/bin/bash -l

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=helmet_ppl ## CHANGE JOBNAME HERE
##SBATCH --array=0

# Remove one # to uncommment
#SBATCH --output=./joblog/%j.out                          ## Stdout
#SBATCH --error=./joblog/%j.err                           ## Stderr

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1                                        ##nodes
#SBATCH -n 1                                        ##tasks
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-1:00:00
#SBATCH --gres=gpu:4 --ntasks-per-node=1 -N 1
##SBATCH --constraint=gpu80
##SBATCH -x "della-i12g[1-2],della-i14g[1-20],della-l01g[3-12],della-l06g12"
#SBATCH --account=danqic
##SBATCH -p ailab
#SBATCH --chdir=/scratch/gpfs/DANQIC/hyen/HELMET
# Turn on mail notification. There are many possible self-explaining values:
# NONE, BEGIN, END, FAIL, ALL (including all aforementioned)
# For more values, check "man sbatch"
#SBATCH --mail-type=ALL
# Remember to set your email address here instead of nobody
#SBATCH --mail-user=nobody

module load proxy/default
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=0
export HF_DATASETS_OFFLINE=1

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"
echo "Array Job ID                   = $SLURM_ARRAY_JOB_ID"
echo "Array Task ID                  = $SLURM_ARRAY_TASK_ID"
echo "Cache                          = $TRANSFORMERS_CACHE"


config=${CONFIG:-"ppl_longmino_64k"}
tag=${TAG:-v1}
seed=${SEED:-42}
model_name=${MODEL_NAME:-Llama-3.1-8B}
model_path=${MODEL_PATH:-/scratch/gpfs/PLI/models/${model_name}}

OPTIONS=${OPTIONS:-""}

echo "Config: ${config}"
echo "Tag: ${tag}"
echo "Seed: ${seed}"
echo "Model name: ${model_name}"
echo "Model path: ${model_path}"
echo "Options: ${OPTIONS}"

uv run python eval_ppl.py \
    --config configs/${config}.yaml \
    --seed ${seed} \
    --output_dir output/${model_name} \
    --tag ${tag} \
    --model_name_or_path ${model_path} $OPTIONS
