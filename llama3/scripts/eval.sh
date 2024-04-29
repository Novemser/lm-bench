weight_importace_dir=./weight_importance/${tasks}
output_dir=./output/${tasks}
echo "output_dir:"${output_dir}
echo "weight_importace_dir:"${weight_importace_dir}
mkdir -p ${output_dir}
mkdir -p ${weight_importace_dir}

# lm_eval --model hf \
#     --model_args \
#         pretrained=${model_path},record_weight_wise_activation=True,output_path=${weight_importace_dir} \
#     --tasks ${tasks}  \
#     --device ${device} \
#     --batch_size auto:1 \
#     --output_path ${output_dir}