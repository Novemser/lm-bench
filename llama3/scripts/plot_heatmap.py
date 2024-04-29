import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

overlap_score_path = 'llama3/output/overlap_score.json'

# 创建一个随机的数据集
data = {
    'A': {'A': 1},
    'B': {'A': 0.2, 'B': 1,},
    'C': {'A': 0.4, 'B': 0.5, 'C': 1}
}
df = pd.DataFrame(data)

# 使用Seaborn库来绘制热力图
sns.heatmap(df, annot=True, cmap='coolwarm', fmt=".1f", linewidths=.5)

# 添加标题
plt.title('Heatmap Example')

plt.savefig('image/heatmap.png')
