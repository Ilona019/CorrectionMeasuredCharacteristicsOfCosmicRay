import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Создание и сохранения журнала логов. Запуск %tensorboard --logdir logs/fit из консоли
def create_fit_logs(log_dir):
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# Сохранить модель нейронной сети в img_file
def save_model_nn(model, img_file):
    tf.keras.utils.plot_model(
        model,
        to_file=img_file,
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=False,
        dpi=96,
        layer_range=None,
        show_layer_activations=True
    )


# Сохранить фигуру matplotlib в файл
def save_plt_figure_to_file(file_dir, format_file='png'):
    plt.savefig(file_dir, format=format_file)
    plt.close()


# Сохранить в табличном виде переданный словарь, где заголовки - ключи
def save_dictionary_to_file(data_dict, file_path):
    for key in data_dict:
        data_dict_key = np.array(data_dict[key])
        shape_key_data = data_dict_key.shape
        if shape_key_data != (-1,):
            data_dict[key] = data_dict_key.reshape(-1,)
    df = pd.DataFrame(data_dict)
    df.to_csv(file_path, sep=' ', index=False, header=True)


# Сохранить текст в файл
def save_text_to_file(file_path, text):
    f = open(file_path, 'w')
    f.write(text)
    f.close()
