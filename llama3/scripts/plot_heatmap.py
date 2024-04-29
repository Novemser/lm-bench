import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json

overlap_score_path = 'output/overlap_score.json'
similarity_path = 'output/similarity.json'

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


def plot_heatmap(path: str, save_path: str, title_prefix: str) -> None:
    all_weight_map = load_data(path)
    for weight_name, weight_map in all_weight_map.items():
        data = weight_map
        df = pd.DataFrame(data)

        # 使用Seaborn库来绘制热力图
        sns.heatmap(df, annot=True, cmap='coolwarm', fmt=".3f", linewidths=.5)

        # 添加标题
        plt.title(title_prefix + weight_name)

        plt.savefig(save_path + '/' + weight_name + '.png')
        plt.clf()

plot_heatmap(overlap_score_path, 'image/overlapscore', 'Overlap Score of ')
plot_heatmap(similarity_path, 'image/similarity', 'Similarity Score of ')