#!/bin/bash -l

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=helmet ## CHANGE JOBNAME HERE
#SBATCH --array=0-35

# Remove one # to uncommment
#SBATCH --output=./joblog/%x-%A_%a.out                          ## Stdout
#SBATCH --error=./joblog/%x-%A_%a.err                           ## Stderr

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1                                        ##nodes
#SBATCH -n 1                                        ##tasks
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=0-24:00:00
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
    IDX=31
    NGPU=1
fi
export OMP_NUM_THREADS=8

# change the tag to distinguish different runs
TAG=v1

CONFIGS=(recall.yaml rag.yaml longqa.yaml summ.yaml icl.yaml rerank.yaml cite.yaml)
SEED=42

OPTIONS=""

M_IDX=$IDX

# Array for models larger than 13B (12 models)
L_MODELS=(
  "Meta-Llama-3-70B-Theta8M"
  "Meta-Llama-3-70B-Instruct-Theta8M"
  "Meta-Llama-3.1-70B"
  "Meta-Llama-3.1-70B-Instruct"
  "Yi-34B-200K"
  "Qwen2-57B-A14B"
  "Qwen2-57B-A14B-Instruct"
  "c4ai-command-r-v01"
  "Jamba-v0.1"
  "AI21-Jamba-1.5-Mini"
  "gemma-2-27b"
  "gemma-2-27b-it"
)

# Array for models 13B and smaller (36 models)
S_MODELS=(
  "LLaMA-2-7B-32K"
  "Llama-2-7B-32K-Instruct"
  "llama-2-7b-80k-basefixed"
  "Yarn-Llama-2-7b-64k"
  "Yarn-Llama-2-7b-128k"
  "Meta-Llama-3-8B"
  "Meta-Llama-3-8B-Instruct"
  "Meta-Llama-3-8B-Theta8M"
  "Meta-Llama-3-8B-Instruct-Theta8M"
  "Meta-Llama-3.1-8B"
  "Meta-Llama-3.1-8B-Instruct"
  "Mistral-7B-v0.1"
  "Mistral-7B-Instruct-v0.1"
  "Mistral-7B-Instruct-v0.2"
  "Mistral-7B-v0.3"
  "Mistral-7B-Instruct-v0.3"
  "Yi-6B-200K"
  "Yi-9B-200K"
  "Yi-1.5-9B-32K"
  "Phi-3-mini-128k-instruct"
  "Phi-3-small-128k-instruct"
  "Phi-3.5-mini-instruct"
  "Qwen2-7B"
  "Qwen2-7B-Instruct"
  "gemma-2-9b"
  "gemma-2-9b-it"
  "prolong-64k-instruct"
  "prolong-512k-instruct-20b-theta128m"
  "Mistral-Nemo-Base-2407"
  "Mistral-Nemo-Instruct-2407"
  "Phi-3-medium-128k-instruct"
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
    # for the base models we always use use_chat_template=False
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

#echo "done, check $OUTPUT_DIR for outputs"

#exit 0

