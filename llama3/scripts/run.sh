source ./scripts/config.sh

tasks="copa"
./scripts/eval.sh

tasks="lambada_openai"
./scripts/eval.sh