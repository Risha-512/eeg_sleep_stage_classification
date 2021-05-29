import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.metrics import accuracy_score, f1_score, jaccard_score, classification_report

from common.edf_parameters import STAGE_NAMES
from common.file_utils import write_to_text_file


def save_classification_report(stage_values: np.array, predicted_stages: np.array, path_to_save: str):
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
