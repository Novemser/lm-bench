import time
import os
from transformers.prune.sparsity_util import task_names as names

task_names = names
timestamp = str(int(time.time()))
topk_percentage = 0.01
model_name = "llama3_hf_weight"
model_name = "Llama-2-7b-chat-hf"
weight_importace_dir = os.path.join("/root/autodl-tmp/weight_importance", model_name)
output_path_prefix = "output"
number_of_bins = 10
aggregate_results = True
reverse_order = False
max_threads = 50