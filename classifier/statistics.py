import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.metrics import accuracy_score, f1_score, jaccard_score, classification_report

from common.edf_parameters import STAGE_NAMES
from common.utils import *

PLOT_X_MAX = 700


def save_classification_report(stage_values: np.array, predicted_stages: np.array, path_to_save: str):
    """
    Сохранить отчет о классификации в файл

    :param stage_values: истинные значения стадий
    :param predicted_stages: предсказанные стадии
    :param path_to_save: путь для сохранения отчета
    """
    f1 = f1_score(stage_values, predicted_stages, average='macro')
    accuracy = accuracy_score(stage_values, predicted_stages)
    jaccard = jaccard_score(stage_values, predicted_stages, average='macro')
    report = classification_report(stage_values, predicted_stages, target_names=STAGE_NAMES.keys(), digits=3)

    write_to_text_file(path_to_save, [
        f'Оценка f1: {f1}',
        f'Оценка точности: {accuracy}',
        f'Коэффициент Жаккара: {jaccard}',
        f'Отчет классификации:\n{report}'
    ])


def save_comparing_plot(stage_values: np.array, predicted_stages: np.array, path_to_save: str):
    """
    Сохранить изображение с графиками гипнограмм

    :param stage_values: истинные значения стадий
    :param predicted_stages: предсказанные стадии
    :param path_to_save: путь для сохранения изображения
    """
    figure, axis = plt.subplots(2, figsize=(12, 10))
    figure.tight_layout(pad=2, h_pad=5)

    x = np.arange(len(stage_values))
    y_ticks = list(STAGE_NAMES.values())
    y_tick_labels = list(STAGE_NAMES.keys())

    values_list = [stage_values, predicted_stages]
    colors = ['tab:green', 'tab:blue']
    titles = ['Исходные стадии', 'Предсказанные стадии']

    for ax, values, color, title in zip(axis, values_list, colors, titles):
        ax.plot(x, values, color=mcolors.TABLEAU_COLORS[color])
        ax.set_title(title)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)
        ax.set_ylabel('Стадии')
        ax.set_xlabel('Время')
        ax.grid()

    plt.savefig(path_to_save)
    figure.clear()
    plt.close(figure)


def save_plots_and_reports(stage_values: np.array,
                           predicted_stages: np.array,
                           file_name: str,
                           plot_dir_path: str,
                           report_dir_path: str):
    """
    Сохранить отчет классификации и изображения гипнограмм

    :param stage_values: истинные значения стадий
    :param predicted_stages: предсказанные стадии
    :param file_name: имя файла
    :param plot_dir_path: путь директории для графиков
    :param report_dir_path: путь директории для отчета
    """
    for idx, chunk in enumerate(split_into_chunks(range(len(stage_values)), chunk_size=PLOT_X_MAX)):
        save_comparing_plot(
            stage_values=stage_values[chunk.start:chunk.stop],
            predicted_stages=predicted_stages[chunk.start:chunk.stop],
            path_to_save=path.join(plot_dir_path, f'{file_name}_[{idx}]{PNG_EXTENSION}')
        )

    save_classification_report(
        stage_values=stage_values,
        predicted_stages=predicted_stages,
        path_to_save=path.join(report_dir_path, f'{file_name}{TXT_EXTENSION}')
    )
