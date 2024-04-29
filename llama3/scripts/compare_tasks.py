import os
import torch
import torch.nn.functional as F
import json
import time

timestamp = str(int(time.time()))
weight_importace_dir = "/root/autodl-tmp/weight_importance"
save_similarity_path = "./output/similarity_" + timestamp + ".json"
save_overlap_path = "./output/overlap_score_" + timestamp + ".json"
topk_percentage = 0.1


def cal_overlap_score(weight1: torch.Tensor, weight2: torch.Tensor, topk_percentage) -> float:
    weight1 = weight1.view(1, -1)
    weight2 = weight2.view(1, -1)
    # get topk most important weights' index
    topk = int(weight1.size(1) * topk_percentage)
    _, index1 = torch.topk(weight1, topk)
    _, index2 = torch.topk(weight2, topk)
    
    mask1 = torch.zeros_like(weight1).scatter(1, index1, 1).bool()
    mask2 = torch.zeros_like(weight2).scatter(1, index2, 1).bool()
    mask = torch.mul(mask1, mask2)
    overlap = mask.sum().item()
    print("weight.size: ", weight1.size())
    print("topk: ", topk)
    print("overlap: ", overlap)
    return overlap / topk

def get_all_weight_name(task: str) -> list[str]:
    task_path = weight_importace_dir + '/' + task
    filenames = os.listdir(task_path)
    return filenames

def compare_two(task1: str, task2: str):
    weight_names = get_all_weight_name(task1)
    weight_names.sort()
    similarity_table = {}
    overlap_table = {}
    for weight_name in weight_names:
        weight_path_1 = weight_importace_dir + '/' + task1 + '/' + weight_name
        weight_path_2 = weight_importace_dir + '/' + task2 + '/' + weight_name
        weight_1 = torch.load(weight_path_1)
        weight_2 = torch.load(weight_path_2)
        similarity = F.cosine_similarity(weight_1.view(1, -1), weight_2.view(1, -1)).item()
        overlap_score = cal_overlap_score(weight_1, weight_2, topk_percentage)
        similarity_table[weight_name] = similarity
        overlap_table[weight_name] = overlap_score
        print(" Weight name: ", weight_name)
        print(" Similarity: ", similarity)
        print(" Overlap score: ", overlap_score)
        
    return similarity_table, overlap_table

begin = time.time()
tasks = ['copa', 'lambada_openai', 'piqa', 'mmlu', 'gsm8k']
cnt = len(tasks)
group_similarity = {}
group_overlap = {}
for i in range(cnt):
    for j in range(i + 1, cnt):
        group_name = tasks[i] + ' vs ' + tasks[j]
        print("Comparing ", group_name)
        group_begin = time.time() 
        group_similarity[group_name], group_overlap[group_name] = compare_two(tasks[i], tasks[j])
        group_end = time.time()
        print("Time: ", group_end - group_begin)
        print("Compare {} finished, total time(s):{}".format(group_name, group_end - group_begin))

end = time.time()
print("Total time(s): ", end - begin)

# save group to json
with open(save_similarity_path, 'w') as json_file:
    json.dump(group_similarity, json_file, indent=4)
with open(save_overlap_path, 'w') as json_file:
    json.dump(group_overlap, json_file, indent=4)

