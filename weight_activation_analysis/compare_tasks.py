import os
import torch
import torch.nn.functional as F
import json
import time
from constants import reverse_order, timestamp, model_name, weight_importace_dir, topk_percentage, task_names
import utils
import concurrent.futures

similarity_file_name = "similarity_" + timestamp + "_" + str(topk_percentage) + ".json"
overlap_file_name = "overlap_score_" + timestamp + "_" + str(topk_percentage) + ".json"

save_similarity_path = os.path.join("output", model_name)
save_overlap_path = os.path.join("output", model_name)

if reverse_order:
    similarity_file_name = "reverse_" + similarity_file_name
    overlap_file_name = "reverse_" + overlap_file_name
    save_similarity_path = os.path.join(save_similarity_path, 'reverse')
    save_overlap_path = os.path.join(save_overlap_path, 'reverse')

utils.create_dir_if_not_exists(save_similarity_path)
utils.create_dir_if_not_exists(save_overlap_path)

def cal_overlap_score(weight1_path: str, weight2_path: str, topk_percentage) -> float:
    weight_1 = torch.load(weight1_path).half()
    weight_2 = torch.load(weight2_path).half()
    weight1 = weight_1.view(1, -1)
    weight2 = weight_2.view(1, -1)
    # get topk most important weights' index
    topk = int(weight1.size(1) * topk_percentage)
    _, index1 = torch.topk(weight1, topk, largest=not reverse_order)
    _, index2 = torch.topk(weight2, topk, largest=not reverse_order)
    
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
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        results = {}
        for weight_name in weight_names:
            weight_path_1 = weight_importace_dir + '/' + task1 + '/' + weight_name
            weight_path_2 = weight_importace_dir + '/' + task2 + '/' + weight_name
            results[weight_name] = executor.submit(cal_overlap_score, weight_path_1, weight_path_2, topk_percentage)
            # similarity = F.cosine_similarity(weight_1.view(1, -1), weight_2.view(1, -1)).item()
            # overlap_score = cal_overlap_score(weight_1, weight_2, topk_percentage)

        for weight_name, future in results.items():
            overlap_score = future.result()
            # similarity_table[weight_name] = similarity
            overlap_table[weight_name] = overlap_score
            print(" Weight name: ", weight_name)
            # print(" Similarity: ", similarity)
            print(" Overlap score: ", overlap_score)
        
    return similarity_table, overlap_table

begin = time.time()
cnt = len(task_names)
group_similarity = {}
group_overlap = {}
for i in range(cnt):
    for j in range(i + 1, cnt):
        group_name = task_names[i] + ' vs ' + task_names[j]
        print("Comparing ", group_name)
        group_begin = time.time() 
        group_similarity[group_name], group_overlap[group_name] = compare_two(task_names[i], task_names[j])
        group_end = time.time()
        print("Time: ", group_end - group_begin)
        print("Compare {} finished, total time(s):{}".format(group_name, group_end - group_begin))

end = time.time()
print("Total time(s): ", end - begin)

# save group to json
# with open(os.path.join(save_similarity_path, similarity_file_name), 'w') as json_file:
#     json.dump(group_similarity, json_file, indent=4)
with open(os.path.join(save_overlap_path, overlap_file_name), 'w') as json_file:
    json.dump(group_overlap, json_file, indent=4)

