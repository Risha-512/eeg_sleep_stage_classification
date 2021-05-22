import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.metrics import accuracy_score, f1_score, classification_report


def show_classification_report(stage_values: np.array, predicted_stages: np.array):
    accuracy = accuracy_score(stage_values, predicted_stages)
    f1 = f1_score(stage_values, predicted_stages, average='macro')

    print(f'Оценка точности: {accuracy}')
    print(f'Оценка f1: {f1}')
    print(f'Отчет классификации:\n{classification_report(stage_values, predicted_stages)}')

    return accuracy, f1


def plot_comparing_graph(raw_values: np.array, stage_values: np.array, predicted_stages: np.array):
    figure, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(14, 12), sharex=True)

    x = np.arange(len(raw_values))

    ax1.plot(x, raw_values, color=mcolors.TABLEAU_COLORS['tab:blue'])
    ax1.set_ylabel('Значения ЭЭГ')
    ax1.set_title('Исходные данные')
    ax1.grid()

    ax2.plot(x, stage_values, color=mcolors.TABLEAU_COLORS['tab:purple'])
    ax2.set_ylabel('Стадии')
    ax2.set_title('Исходные стадии')
    ax2.grid()

    ax3.plot(x, predicted_stages, color=mcolors.TABLEAU_COLORS['tab:cyan'])
    ax3.set_ylabel('Стадии')
    ax3.set_title('Предсказанные стадии')
    ax3.grid()

    ax4.plot(x, stage_values, label='true values', color=mcolors.TABLEAU_COLORS['tab:purple'])
    ax4.plot(x, predicted_stages, label='predicted values', color=mcolors.TABLEAU_COLORS['tab:cyan'])
    ax4.set_ylabel('Стадии')
    ax4.set_title('Сравнение')
    ax4.legend(loc='upper left')
    ax4.grid()

    plt.xlabel('Время')
    plt.show()


