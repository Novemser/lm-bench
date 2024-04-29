outputDir="./output/lambada_openai"
echo ${outputDir}
mkdir -p ${outputDir}
tasks="lambada_openai"
model_path="./llama3_hf_weights"
lm_eval --model hf \
    --model_args \
        pretrained=${model_path},dtype="float32",record_weight_wise_activation=True,output_path=${outputDir} \
    --tasks ${tasks}  \
    --device cpu \
    --batch_size auto:1 \
    --output_path ${outputDir}