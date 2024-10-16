#!/bin/bash -l

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=helmet_short ## CHANGE JOBNAME HERE
#SBATCH --array=0

# Remove one # to uncommment
#SBATCH --output=./joblog/%x-%A_%a.out                          ## Stdout
#SBATCH --error=./joblog/%x-%A_%a.err                           ## Stderr

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1                                        ##nodes
#SBATCH -n 1                                        ##tasks
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=0-4:00:00
#SBATCH --gres=gpu:1 --ntasks-per-node=1 -N 1
#SBATCH --constraint=gpu80
# Turn on mail notification. There are many possible self-explaining values:
# NONE, BEGIN, END, FAIL, ALL (including all aforementioned)
# For more values, check "man sbatch"
#SBATCH --mail-type=ALL
# Remember to set your email address here instead of nobody
#SBATCH --mail-user=nobody

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
    IDX=0
    NGPU=1
fi
PORT=$(shuf -i 30000-65000 -n 1)
echo "Port                          = $PORT"

export OMP_NUM_THREADS=8

TAG=v1

CONFIGS=(recall_short.yaml rag_short.yaml longqa_short.yaml summ_short.yaml icl_short.yaml rerank_short.yaml cite_short.yaml)
#CONFIGS=(${CONFIGS[8]})
SEED=42

M_IDX=$IDX

# Array for models larger than 13B (12 models)
L_MODELS=(
  "Meta-Llama-3-70B-Theta8M" #0
  "Meta-Llama-3-70B-Instruct-Theta8M" #1
  "Meta-Llama-3.1-70B" #2
  "Meta-Llama-3.1-70B-Instruct" #3
  "Yi-34B-200K" #4
  "Qwen2-57B-A14B" #5
  "Qwen2-57B-A14B-Instruct" #6
  "c4ai-command-r-v01" #7
  "Jamba-v0.1" #8
  "AI21-Jamba-1.5-Mini" #9
  "gemma-2-27b" #10
  "gemma-2-27b-it" #11
)

# Array for models 13B and smaller (36 models)
S_MODELS=(
  "LLaMA-2-7B-32K" # 0
  "Llama-2-7B-32K-Instruct" # 1
  "llama-2-7b-80k-basefixed" # 2
  "Yarn-Llama-2-7b-64k" # 3
  "Yarn-Llama-2-7b-128k" # 4
  "Meta-Llama-3-8B" # 5
  "Meta-Llama-3-8B-Instruct" # 6
  "Meta-Llama-3-8B-Theta8M" # 7
  "Meta-Llama-3-8B-Instruct-Theta8M" # 8
  "Meta-Llama-3.1-8B" # 9
  "Meta-Llama-3.1-8B-Instruct" # 10
  "Mistral-7B-v0.1" # 11
  "Mistral-7B-Instruct-v0.1" # 12
  "Mistral-7B-Instruct-v0.2" # 13
  "Mistral-7B-v0.3" # 14
  "Mistral-7B-Instruct-v0.3" # 15
  "Yi-6B-200K" # 16
  "Yi-9B-200K" # 17
  "Yi-1.5-9B-32K" # 18
  "Phi-3-mini-128k-instruct" # 19
  "Phi-3-small-128k-instruct" # 20
  "Phi-3.5-mini-instruct" # 21
  "Qwen2-7B" # 22
  "Qwen2-7B-Instruct" # 23
  "gemma-2-9b" # 24
  "gemma-2-9b-it" # 25
  "prolong-64k-instruct" # 26
  "prolong-512k-instruct-20b-theta128m" # 27
  "Mistral-Nemo-Base-2407" # 28
  "Mistral-Nemo-Instruct-2407" # 29
  "Phi-3-medium-128k-instruct" # 30
  "MegaBeam-Mistral-7B-512k" #31
  "Llama-3.2-1B" # 32
  "Llama-3.2-1B-Instruct" # 33
  "Llama-3.2-3B" # 34
  "Llama-3.2-3B-Instruct" # 35
)
MNAME="${S_MODELS[$M_IDX]}"

OUTPUT_DIR="output/$MNAME"
MODEL_NAME="/path/to/your/model/$MNAME" # CHANGE PATH HERE or you can change the array to load from HF

shopt -s nocasematch
chat_models=".*(chat|instruct|it$|nous|command|Jamba-1.5|MegaBeam).*"
echo $MNAME
if ! [[ $MNAME =~ $chat_models ]]; then
    OPTIONS="$OPTIONS --use_chat_template False"
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
