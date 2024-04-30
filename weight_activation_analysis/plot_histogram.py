import os
import matplotlib.pyplot as plt
from constants import model_name, weight_importace_dir, output_path_prefix, task_names, aggregate_results, number_of_bins
import utils as utils
import torch
import seaborn as sns
import pandas as pd
import logging
import numpy as np

histogram_figure_path = os.path.join(output_path_prefix, model_name, 'weight_activation_histogram')
utils.create_dir_if_not_exists(histogram_figure_path)

def random_sample(arr: np.array, size: int = 1) -> np.array:
    return arr[np.random.choice(len(arr), size=size, replace=False)]

def plot_weight_activation(weight: torch.tensor, weight_name: str):
    logging.warning("Plotting CDF for {}".format(weight_name))
    weights = weight.view(1, -1).cpu()[0].numpy()
    # weights = random_sample(weights, 10000)
    # weights = np.sort(weights)
    # weights = weights / weights.max()
    plt.figure(figsize=(15, 5))
    # plt.bar(np.arange(0, len(weights)), weights)
    sns.ecdfplot(data=pd.DataFrame(weights))
    plt.savefig(os.path.join(histogram_figure_path, weight_name.replace(".pt", ".png")))
    plt.clf()

def draw_histogram_of_task_weight_activations(task_name: str):
    weight_names = utils.get_all_weight_name(task_name)
    for weight_name in weight_names:
        weight_path = os.path.join(weight_importace_dir, task_name, weight_name)
        weight = torch.load(weight_path).half()
        plot_weight_activation(weight, weight_name)
        
if __name__ == '__main__':
    for task_name in task_names:
        draw_histogram_of_task_weight_activations(task_name)