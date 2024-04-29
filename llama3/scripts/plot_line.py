import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json

overlap_score_path = 'output/overlap_score.json'
similarity_path = 'output/similarity.json'

layer_num = 32
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
# marker = ['o', 'v', '^', '<', '>', 's', 'p']

def get_all_weight_name(data) -> set:
    weight_names = set()
    for task_pair, group_table in data.items():
        for weight_name, score in group_table.items():
            weight_name = weight_name.split('_', 1)[1]
            weight_names.add(weight_name)
    return weight_names
            
def plot_line_internal(layer_scores, task_pair, weight_name):
    assert len(layer_scores) == layer_num
    plt.plot(range(0, layer_num), layer_scores)

def plot_line(path: str):
    with open(path, 'r') as f:
        data = json.load(f)
    weight_names = get_all_weight_name(data)
    for task_pair, group_table in data.items():
        plt.figure(figsize=(15, 5))
        for idx, weight_name in enumerate(weight_names):
            layer_scores = []
            for i in range(0, layer_num):
                key = f'{i}_{weight_name}'
                layer_scores.append(group_table[key])
            plt.plot(range(0, layer_num), layer_scores, linestyle='--', label=weight_name, color=colors[idx], marker='o')
        plt.xlabel('Layer id')
        plt.ylabel('Score')
        plt.title(f'{task_pair}')
        plt.legend()
        plt.savefig(f'image/overlapscore/line/{task_pair}.png')
        plt.clf()
    
plot_line(overlap_score_path)