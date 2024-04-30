export PYTHONPATH=/root/modelbase/lm-bench/llm-eval-harness/transformers/src
export tasks="copa"
export device=cuda
export MODEL_NAME=llama3_hf_weight
export model_path=/root/autodl-tmp/${MODEL_NAME}
# export model_path=meta-llama/Llama-2-13b-hf
export HF_ENDPOINT=https://hf-mirror.com
source /root/tool/http_proxy.sh