#!/bin/bash -l

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=helmet ## CHANGE JOBNAME HERE
#SBATCH --array=0-12

# Remove one # to uncommment
#SBATCH --output=./joblog/%x-%A_%a.out                          ## Stdout
#SBATCH --error=./joblog/%x-%A_%a.err                           ## Stderr

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1                                        ##nodes
#SBATCH -n 1                                        ##tasks
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=0-16:00:00
#SBATCH --gres=gpu:4 --ntasks-per-node=1 -N 1
#SBATCH --constraint=gpu80
# Turn on mail notification. There are many possible self-explaining values:
# NONE, BEGIN, END, FAIL, ALL (including all aforementioned)
# For more values, check "man sbatch"
#SBATCH --mail-type=ALL
# Remember to set your email address here instead of nobody
#SBATCH --mail-user=nobody

module load proxy/default

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

source env/bin/activate

IDX=$SLURM_ARRAY_TASK_ID
NGPU=$SLURM_GPUS_ON_NODE
if [[ -z $SLURM_ARRAY_TASK_ID ]]; then
    IDX=5
    NGPU=0
fi
export OMP_NUM_THREADS=8
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# change the tag to distinguish different runs
TAG=v1

CONFIGS=(recall_add.yaml recall_add_long.yaml rag.yaml icl.yaml rerank.yaml cite.yaml mrcr.yaml mrcr_long.yaml longbenchv2.yaml longbenchv2_long.yaml)
#CONFIGS=(mrcr.yaml mrcr_long.yaml longbenchv2.yaml longbenchv2_long.yaml)
#CONFIGS=(mrcr.yaml mrcr_long.yaml longbenchv2.yaml longbenchv2_long.yaml)
SEED=42

OPTIONS="--use_vllm"

M_IDX=$IDX


MODELS=(
  "Meta-Llama-3-8B-Instruct"
  "Llama-3.1-8B-Instruct"
  "Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct"
  "Llama-3.1-Nemotron-8B-UltraLong-2M-Instruct"
  "Llama-3.1-Nemotron-8B-UltraLong-4M-Instruct"
  "gemma-3-4b-it"
  "Llama-3-8B-ProLong-512k-Instruct"
  "Qwen2.5-7B-Instruct"
  "Qwen2.5-7B-Instruct-1M"
  "Kimi-Linear-48B-A3B-Instruct"
  "Qwen3-30B-A3B-Instruct-2507"
  "NVIDIA-Nemotron-Nano-9B-v2"
  "gpt-oss-120b"
)


MNAME="${MODELS[$M_IDX]}"

OUTPUT_DIR="output/$MNAME"
MODEL_NAME="/scratch/gpfs/PLI/models/$MNAME" # CHANGE PATH HERE or you can change the array to load from HF

shopt -s nocasematch
chat_models=".*(chat|instruct|it$|nous|command|Jamba-1.5|MegaBeam).*"
echo $MNAME
if ! [[ $MNAME =~ $chat_models ]]; then
    # for the base models we always use use_chat_template=False
    OPTIONS="$OPTIONS --use_chat_template False"
fi


if [[ $MNAME =~ ".*(Qwen2.5-7B-Instruct|Qwen3).*" ]]; then
    # for the base models we always use use_chat_template=False
    OPTIONS="$OPTIONS --rope_scaling 4"
fi


echo "Evaluation output dir         = $OUTPUT_DIR"
echo "Tag                           = $TAG"
echo "Model name                    = $MODEL_NAME"
echo "Options                       = $OPTIONS"


for CONFIG in "${CONFIGS[@]}"; do
    echo "Config file: $CONFIG"

    python eval.py \
        --config configs/$CONFIG \
        --seed $SEED \
        --output_dir $OUTPUT_DIR \
        --tag $TAG \
        --model_name_or_path $MODEL_NAME \
        $OPTIONS
done

echo "finished with $?"

wait;

#echo "done, check $OUTPUT_DIR for outputs"

#exit 0

