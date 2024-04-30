import os
import matplotlib.pyplot as plt
from constants import model_name, weight_importace_dir, output_path_prefix, task_names, aggregate_results
import utils as utils
import torch
import seaborn as sns

histogram_figure_path = os.path.join(output_path_prefix, model_name, 'weight_activation_histogram')
utils.create_dir_if_not_exists(histogram_figure_path)

def plot_weight_activation(weight: torch.tensor, weight_name: str):
    sorted_weights = weight.view(1, -1).sort().values.cpu()[0].numpy()
    sns.ecdfplot(data=sorted_weights, cumulative=True, x="CDF of weight activation of {}".format(weight_name))
    plt.savefig(os.path.join(histogram_figure_path, weight_name))

def draw_histogram_of_task_weight_activations(task_name: str):
    weight_names = utils.get_all_weight_name(task_name)
    for weight_name in weight_names:
        weight_path = os.path.join(weight_importace_dir, task_name, weight_name)
        weight = torch.load(weight_path).half()
        plot_weight_activation(weight)
        
if __name__ == '__main__':
    for task_name in task_names:
        draw_histogram_of_task_weight_activations(task_name)