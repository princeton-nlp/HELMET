
LLM_ENDPOINT="https://${hf_inference_point_url}/v1" # fill in your endpoint url
API_KEY=$HF_TOKEN

python eval.py --config configs/recall_demo.yaml --endpoint_url $LLM_ENDPOINT --api_key $API_KEY