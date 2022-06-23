import numpy as np
import mglearn as mgl


# Создает тепловую карту
def create_heatmap(df_column, name_xlabel, name_ylabel, xticklabels, yticklabels, cmap, format):
    scores = np.array(df_column).reshape(len(yticklabels), len(xticklabels))
    mgl.tools.heatmap(scores, xlabel=name_xlabel, ylabel=name_ylabel, xticklabels=xticklabels,
                      yticklabels=yticklabels, cmap=cmap, fmt=format)