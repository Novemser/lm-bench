import os

def get_all_weight_name(weight_importace_dir: str, task: str) -> list[str]:
    task_path = os.path.join(weight_importace_dir, task)
    return os.listdir(task_path).sort()

def create_dir_if_not_exists(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)
