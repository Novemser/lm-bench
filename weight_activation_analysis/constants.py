import time
import os

task_names = ['copa', 'lambada_openai', 'piqa', 'mmlu', 'gsm8k', 'arc_challenge']
timestamp = str(int(time.time()))
topk_percentage = 0.01
model_name = "llama3_hf_weight"
weight_importace_dir = os.path.join("/root/autodl-tmp/weight_importance", model_name)
output_path_prefix = "output"
number_of_bins = 500