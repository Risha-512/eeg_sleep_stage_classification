import numpy as np

from os import path, pardir
from random import choice
from dataclasses import dataclass
from progressbar import progressbar
from sklearn.model_selection import train_test_split

from typing import Generator, Sized
from argparse import ArgumentParser
from tensorflow.python.keras.models import Model

from model import ModelCNN, ModelCallbacks
from statistics import save_classification_report, save_comparing_plot
from common.file_utils import *
from common.npz_parameters import *
from common.edf_parameters import *

H5_EXTENSION = '.h5'
TXT_EXTENSION = '.txt'
PNG_EXTENSION = '.png'
DEFAULT_FILE_NAME = '(0_8__0_87)'

DEFAULT_MODEL_FILE_PATH = path.join(pardir, 'models', DEFAULT_FILE_NAME + H5_EXTENSION)
DEFAULT_REPORT_FILE_PATH = path.join(pardir, 'reports', DEFAULT_FILE_NAME + TXT_EXTENSION)
DEFAULT_PLOT_DIR_PATH = path.join(pardir, 'plots', DEFAULT_FILE_NAME)

WINDOW_SIZE = 100


def parse_arguments():
    """
    Получить входные аргументы программы:
        - (--input_directory) путь к директории с исходными данными
        - (--model_file_path) путь к файлу модели сверточной нейронной сети
        - (--report_file_path) путь к файлу с отчетом по классификации
        - (--plot_dir_path) путь к директории изображений графиков
        - (--do_fit) параметр, определяющий, требуется ли обучение, или только загрузка весов из файла
    """
    parser = ArgumentParser()

    parser.add_argument('--input_directory', type=str, default=NPZ_DIRECTORY_PATH,
                        help='Путь к файлам npz')
    parser.add_argument('--model_file_path', type=str, default=DEFAULT_MODEL_FILE_PATH,
                        help='Путь к файлу модели')
    parser.add_argument('--report_file_path', type=str, default=DEFAULT_REPORT_FILE_PATH,
                        help='Путь к файлу с отчетом по классификации')
    parser.add_argument('--plot_dir_path', type=str, default=DEFAULT_PLOT_DIR_PATH,
                        help='Путь к директории изображений графиков')
    parser.add_argument('--do_fit', type=bool, default=False,
                        help='True, если требуется обучение, иначе только загрузка модели из файла')
    return parser.parse_args()


def train_test_validation_split(data: Sized) -> (Sized, Sized, Sized):
    """
    Разделить список данных на три списка:
        - список данных для обучения
        - список данных для тестирования
        - список данных для валиадции (проверки)

    :param data: список данных, который нужно разделить
    :return: список данных для обучения, список данных для тестирования, список данных для валидации
    """
    train_val, test = train_test_split(data, test_size=0.15, random_state=1338)
    train, validation = train_test_split(train_val, test_size=0.1, random_state=1337)

    return train, test, validation


def load_npz_files(npz_paths: List[str]) -> dict:
    """
    Загрузить все файлы npz из списка

    :param npz_paths: список путей файлов npz
    :return: словарь из загруженных файлов
    """
    return {npz_path: np.load(npz_path) for npz_path in npz_paths}


def rescale_and_clip_array(array: np.array, scale: int = 0.05):
    """
    Масштабировать значения массива и убрать излишние

    :param array: массив для масштабирования
    :param scale: масштаб
    :return: масштабированный массив
    """
    array = array * scale
    array = np.clip(array, -scale * 100, scale * 100)
    return array


def data_to_generator(data: dict, count: int = 10, window_size: int = WINDOW_SIZE) -> Generator:
    """
    Преобразовать данные в генератор из значений ЭЭГ и соответствующих стадий

    :param data: словарь данных для преобразования
    :param count: количество итераций генератора
    :param window_size: размер блока данных
    :return: значения ЭЭГ и соответствующие стадии
    """
    while True:
        chosen_data = data[choice(list(data.keys()))]

        assert len(chosen_data[RAW_VALUES_KEY]) == len(chosen_data[STAGE_VALUES_KEY])
        size = len(chosen_data[RAW_VALUES_KEY]) - window_size

        for i in range(count):
            idx = choice(range(size))
            raw_values = chosen_data[RAW_VALUES_KEY][idx:idx + window_size]
            stage_values = chosen_data[STAGE_VALUES_KEY][idx:idx + window_size]

            # увеличить размерность вектора и масштабировать его значения
            raw_values = prepare_raw_values_for_model(raw_values)

            # транспонировать вектор и увеличить размерность на 1
            stage_values = np.expand_dims(stage_values, -1)
            stage_values = np.expand_dims(stage_values, 0)

            yield raw_values, stage_values


def split_into_chunks(data: Sized, chunk_size: int) -> list:
    """
    Разделить данные на равные блоки

    :param data: данные, которые необходимо поделить
    :param chunk_size: размер блока
    :return: список из разделенных на блоки данных
    """
    return [data[idx:idx + chunk_size] for idx in range(0, len(data), chunk_size)]


def prepare_raw_values_for_model(array: np.array):
    """
    Подготовить массив к обработке моделью:
        - изменить размерность
        - масштабировать
        - исключить лишние значения

    :param array: изменяемый массив
    :return: измененный массив
    """
    assert len(array.shape) <= 4

    # увеличивать размерность на 1, пока не достигнет 4
    while len(array.shape) != 4:
        array = np.expand_dims(array, 0)

    return rescale_and_clip_array(array)


def convert_array_to_1d_list(array: np.array) -> list:
    """
    Преобразовать numpy массив в одномерный список

    :param array: преобразуемый numpy массив
    :return: список, полученный из входного массива
    """
    return array.ravel().tolist()


@dataclass
class StagePredictionData:
    raw_values: List[float]
    stage_values: List[int]
    predicted_stages: List[int]


def predict_stages(model: Model, test_data: dict) -> StagePredictionData:
    """
    Предсказать стадии на основе тестовых данных

    :param model: модель, выполняющая предсказание
    :param test_data: тестовые данные, на основе которых делается предсказание
    :return: список верных данных и список соответствующих предсказанных данных
    """
    raw_values_1d, stage_values_1d, predicted_stages = [], [], []

    for test_data_key in progressbar(test_data):
        raw_values = test_data[test_data_key][RAW_VALUES_KEY]
        stage_values = test_data[test_data_key][STAGE_VALUES_KEY]

        assert len(raw_values) == len(stage_values)

        raw_values = prepare_raw_values_for_model(raw_values)

        raw_values_1d += convert_array_to_1d_list(raw_values)
        stage_values_1d += convert_array_to_1d_list(stage_values)
        predicted_stages += convert_array_to_1d_list(model.predict(raw_values).argmax(axis=-1))

    return StagePredictionData(raw_values_1d, stage_values_1d, predicted_stages)


def main():
    args = parse_arguments()

    # получить пути всех файлов данных и разделить их на данные для обучения, тестирования и валидации
    npz_files = get_files_in_directory(args.input_directory, NPZ_FILE_PATTERN)
    train_files, test_files, validation_files = train_test_validation_split(npz_files)

    # загрузить данные из файлов
    train_data, test_data, validation_data = (
        load_npz_files(train_files),
        load_npz_files(test_files),
        load_npz_files(validation_files)
    )

    # создать экземпляр модели сверточной нейронной сети
    model = ModelCNN(STAGES_TYPES_NUMBER).generate_cnn_model()

    # обучить модель, если передан соответствующий параметр
    if args.do_fit:
        model.fit(
            data_to_generator(train_data),
            validation_data=data_to_generator(validation_data),
            epochs=100,
            verbose=2,
            steps_per_epoch=1000,
            validation_steps=300,
            callbacks=ModelCallbacks(args.model_file_path).generate_model_callbacks()
        )
    # загрузить модель из файла
    model.load_weights(args.model_file_path)

    # предсказать стадии на основе тестовых данных
    prediction = predict_stages(model, test_data)

    # оценить предсказание
    save_classification_report(
        stage_values=prediction.stage_values,
        predicted_stages=prediction.predicted_stages,
        path_to_save=args.report_file_path)

    for idx, chunk in enumerate(split_into_chunks(range(len(prediction.stage_values)), chunk_size=700)):
        save_comparing_plot(
            stage_values=prediction.stage_values[chunk.start:chunk.stop],
            predicted_stages=prediction.predicted_stages[chunk.start:chunk.stop],
            path_to_save=f'{args.plot_dir_path}_[{idx}]{PNG_EXTENSION}'
        )


if __name__ == '__main__':
    main()
