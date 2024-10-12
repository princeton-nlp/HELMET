#!/bin/bash -l

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=api ## CHANGE JOBNAME HERE
#SBATCH --array=0

# Remove one # to uncommment
#SBATCH --output=./joblog/%x-%A_%a.out                          ## Stdout
#SBATCH --error=./joblog/%x-%A_%a.err                           ## Stderr

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1                                        ##nodes
#SBATCH -n 1                                        ##tasks
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-3:00:00
#SBATCH --gres=gpu:0 --ntasks-per-node=1 -N 1
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

export OMP_NUM_THREADS=8
IDX=$SLURM_ARRAY_TASK_ID
if [[ -z $SLURM_ARRAY_TASK_ID ]]; then
    IDX=0
fi


TAG=v1

CONFIGS=(recall.yaml rag.yaml longqa.yaml summ.yaml icl.yaml rerank.yaml cite.yaml)
#CONFIGS=(${CONFIGS[7]}) # you may want to run only one config
SEED=42

# azure vs. non-azure makes no difference, just use whichever you prefer
OD=(
    azure/gpt-4-0125-preview # 0
    azure/gpt-4o-2024-05-13 # 1
    gpt-4o-2024-08-06 # 2
    azure/gpt-4o-mini-2024-07-18  # 3
    claude-3-5-sonnet-20240620 # 4
    gemini-1.5-flash-001 # 5
    gemini-1.5-pro-001 # 6
)
MODEL_NAME="${OD[$IDX]}"
OUTPUT_DIR="output/$(basename $MODEL_NAME)"

# for the API models we always use use_chat_template=True
OPTIONS="--use_chat_template True --stop_newline False"

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
