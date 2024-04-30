source ./scripts/config.sh

evaluate_all() {
    tasks="copa"
    ./scripts/eval.sh

    tasks="lambada_openai"
    ./scripts/eval.sh

    tasks="piqa"
    ./scripts/eval.sh

    tasks="mmlu"
    ./scripts/eval.sh

    tasks="gsm8k"
    ./scripts/eval.sh

    tasks="arc_challenge"
    ./scripts/eval.sh
}
# export MODEL_NAME=Llama-2-7b-chat-hf
# export model_path=meta-llama/Llama-2-7b-chat-hf
# evaluate_all

export MODEL_NAME=Llama-2-13b-chat-hf
export model_path=meta-llama/Llama-2-13b-chat-hf
evaluate_all