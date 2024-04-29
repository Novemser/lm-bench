import os
import torch
import torch.nn.functional as F
import json

weight_importace_dir = "/root/autodl-tmp/weight_importance"
save_path = "./output/similarity.json"

def get_all_weight_name(task: str) -> list[str]:
    task_path = weight_importace_dir + '/' + task
    filenames = os.listdir(task_path)
    return filenames

def compare_two(task1: str, task2: str):
    weight_names = get_all_weight_name(task1)
    weight_names.sort()
    group = {}
    for weight_name in weight_names:
        weight_path_1 = weight_importace_dir + '/' + task1 + '/' + weight_name
        weight_path_2 = weight_importace_dir + '/' + task2 + '/' + weight_name
        weight_1 = torch.load(weight_path_1)
        weight_2 = torch.load(weight_path_2)
        similarity = F.cosine_similarity(weight_1.view(1, -1), weight_2.view(1, -1))
        group[weight_name] = similarity.item()
        
    return group

tasks = ['copa', 'lambada_openai', 'piqa', 'mmlu']
cnt = len(tasks)
group = {}
for i in range(cnt):
    for j in range(i + 1, cnt):
        group_name = tasks[i] + '-' + tasks[j]
        group[group_name] = compare_two(tasks[i], tasks[j])

# save group to json
with open(save_path, 'w') as json_file:
    json.dump(group, json_file, indent=4)

