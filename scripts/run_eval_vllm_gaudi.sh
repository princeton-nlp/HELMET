export host_ip=$(hostname -I | awk '{print $1}')
export LLM_ENDPOINT_PORT=8010
export DATA_PATH="~/.cache/huggingface"
export LLM_ENDPOINT="http://${host_ip}:${LLM_ENDPOINT_PORT}/v1"
export HF_HOME=$DATA_PATH

for task in "recall" "rag"; do
    python eval.py --config configs/${task}_vllm.yaml --endpoint_url $LLM_ENDPOINT --overwrite --batch_mode multi_thread --no_cuda
done