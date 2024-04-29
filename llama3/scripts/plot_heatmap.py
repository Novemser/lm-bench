import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json

overlap_score_path = 'output/overlap_score_1714395810_0.01.json'
similarity_path = 'output/similarity_1714395810_0.01.json'

def load_data(path: str):
    with open(path, 'r') as f:
        data = json.load(f)
    
    preprocessed_data = {}
    for task_pair, group_table in data.items():
        task1, task2 = task_pair.split('vs')
        task1.strip()
        task2.strip()
        print(task1, task2)
        for weight_name, score in group_table.items():
            if weight_name not in preprocessed_data:
                preprocessed_data[weight_name] = {}
            if task1 not in preprocessed_data[weight_name]:
                preprocessed_data[weight_name][task1] = {}
            preprocessed_data[weight_name][task1][task2] = score
    
    return preprocessed_data    


def plot_heatmap(path: str, save_path: str, title_prefix: str, aggregate=True) -> None:
    all_weight_map = load_data(path)
    if aggregate:
        aggregated_weight_map = {}
        aggregated_weight_map_counter = {}
        for weight_name, weight_map in all_weight_map.items():
            # replace the numbers at the begining
            weight_name = weight_name[weight_name.index('_') + 1:]
            if weight_name not in aggregated_weight_map:
                aggregated_weight_map[weight_name] = {}
                aggregated_weight_map_counter[weight_name] = {}
            for task1, tasks in weight_map.items():
                if task1 not in aggregated_weight_map[weight_name]:
                    aggregated_weight_map[weight_name][task1] = {}
                    aggregated_weight_map_counter[weight_name][task1] = {}
                for task2, score in tasks.items():
                    if task2 not in aggregated_weight_map[weight_name][task1]:
                        aggregated_weight_map[weight_name][task1][task2] = 0.0
                        aggregated_weight_map_counter[weight_name][task1][task2] = 0
                    aggregated_weight_map[weight_name][task1][task2] += score
                    aggregated_weight_map_counter[weight_name][task1][task2] += 1
        
        for agg_weight_name, agg_weight_map in aggregated_weight_map.items():
            for task1, tasks in agg_weight_map.items():
                for task2, total_score in tasks.items():
                    aggregated_weight_map[agg_weight_name][task1][task2] /= \
                        aggregated_weight_map_counter[agg_weight_name][task1][task2]
        all_weight_map = aggregated_weight_map
    for weight_name, weight_map in all_weight_map.items():
        data = weight_map
        df = pd.DataFrame(data)

        # 使用Seaborn库来绘制热力图
        sns.heatmap(df, annot=True, cmap='coolwarm', fmt=".3f", linewidths=.5)

        # 添加标题
        plt.title(title_prefix + weight_name)

        plt.savefig(save_path + '/' + weight_name + '.png')
        plt.clf()

plot_heatmap(overlap_score_path, 'image/aggerage/overlapscore', 'Overlap Score of ')
plot_heatmap(similarity_path, 'image/aggerage/similarity', 'Similarity Score of ')