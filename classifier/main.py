import numpy as np

from random import choice
from progressbar import progressbar
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from typing import Generator, Sized
from argparse import ArgumentParser
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.engine.functional import Functional

from model_cnn import ModelCNN
from common.file_utils import *
from common.npz_parameters import *
from common.edf_parameters import STAGES_TYPES_NUMBER

MODEL_FILE_PATH = '..\\models\\model_(0_8__0_87).h5'

CHUNK_SIZE = 100


def parse_arguments():
    """
    Получить входные аргументы программы:
        - (--input_directory) путь к директории с иходными данными
        - (--model_file_path) путь к файлу модели сверточной нейронной сети
        - (--do_fit) параметр, определяющий, требуется ли обучение, или только загрузка весов из файла
    """
    parser = ArgumentParser()
    parser.add_argument('--input_directory', type=str, default=NPZ_DIRECTORY_PATH, help='Путь к файлам npz')
    parser.add_argument('--model_file_path', type=str, default=MODEL_FILE_PATH, help='Путь к файлу модели')
    parser.add_argument('--do_fit', type=bool, default=False,
                        help='True, если требуется обучение, иначе только загрузка модели из файла')
    return parser.parse_args()


def train_test_validation_split(data: list) -> (list, list, list):
    """
    Разделить список данных на три:
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


def data_to_generator(data: dict, count: int = 10, chunk_size: int = CHUNK_SIZE) -> Generator:
    """
    Преобразовать данные в генератор из значений ЭЭГ и соответствующих стадий

    :param data: словарь данных для преобразования
    :param count: количество итераций генератора
    :param chunk_size: размер блока данных
    :return: значения ЭЭГ и соответствующие стадии
    """
    while True:
        chosen_data = data[choice(list(data.keys()))]

        assert len(chosen_data[RAW_VALUES_KEY]) == len(chosen_data[STAGE_VALUES_KEY])
        size = len(chosen_data[RAW_VALUES_KEY]) - chunk_size

        for i in range(count):
            idx = choice(range(size))

            raw_values = chosen_data[RAW_VALUES_KEY][idx:idx + chunk_size]
            raw_values = np.expand_dims(raw_values, 0)
            raw_values = rescale_and_clip_array(raw_values)

            stage_values = chosen_data[STAGE_VALUES_KEY][idx:idx + chunk_size]
            stage_values = np.expand_dims(stage_values, -1)
            stage_values = np.expand_dims(stage_values, 0)

            yield raw_values, stage_values


def generate_model_callbacks(model_file_path: str) -> list:
    """
    Сгенерировать список экземпляров обратных вызовов, применяемых во время обучения
        - ModelCheckpoint - сохраняет модель (или веса) в файл контрольной точки
        - EarlyStopping - прекращает обучение, когда отслеживаемый показатель перестал улучшаться
        - ReduceLROnPlateau - уменьшить скорость обучения, когда отслеживаемый показатель перестал улучшаться

    :param model_file_path: путь файла данных модели
    :return: список экземпляров обратных вызовов
    """
    checkpoint = ModelCheckpoint(model_file_path, monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=10)
    reduce_learning_rate = ReduceLROnPlateau(monitor='val_acc', mode='max', verbose=1, patience=5)

    return [checkpoint, early_stopping, reduce_learning_rate]


def split_into_chunks(data: Sized, chunk_size: int = CHUNK_SIZE) -> list:
    """
    Разделить данные на равные блоки

    :param data: данные, которые необходимо поделить
    :param chunk_size: размер блока
    :return: список из разделенных на блоки данных
    """
    return [data[idx:idx + chunk_size] for idx in range(0, len(data), chunk_size)]


def prepare_array_for_prediction(array: np.array):
    """
    Подготовить массив к предсказанию:
        - изменить размерность
        - масштабировать
        - исключить лишние значения

    :param array: изменяемый массив
    :return: измененный массив
    """
    assert len(array.shape) <= 4

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


def predict_stages(model: Functional, test_data: dict) -> (list, list):
    """
    Предсказать стадии на основе тестовых данных

    :param model: модель, выполняющая предсказание
    :param test_data: тестовые данные, на основе которых делается предсказание
    :return: список верных данных и список соответствующих предсказанных данных
    """
    true_stages, predicted_stages = [], []

    for test_data_key in progressbar(test_data):
        raw_values = test_data[test_data_key][RAW_VALUES_KEY]
        stage_values = test_data[test_data_key][STAGE_VALUES_KEY]

        assert len(raw_values) == len(stage_values)

        for chunk in split_into_chunks(range(len(raw_values))):
            raw_values_chunk = prepare_array_for_prediction(raw_values[chunk.start:chunk.stop])

            true_stages += convert_array_to_1d_list(stage_values[chunk.start:chunk.stop])
            predicted_stages += convert_array_to_1d_list(model.predict(raw_values_chunk).argmax(axis=-1))

    return true_stages, predicted_stages


def main():
    args = parse_arguments()

    npz_files = get_files_in_directory(args.input_directory, NPZ_FILE_PATTERN)
    train_files, test_files, validation_files = train_test_validation_split(npz_files)

    train_data, test_data, validation_data = (
        load_npz_files(train_files),
        load_npz_files(test_files),
        load_npz_files(validation_files)
    )

    model = ModelCNN(STAGES_TYPES_NUMBER).generate_cnn_model()

    if args.do_fit:
        model.fit(
            data_to_generator(train_data),
            validation_data=data_to_generator(validation_data),
            epochs=100,
            verbose=2,
            steps_per_epoch=1000,
            validation_steps=300,
            callbacks=generate_model_callbacks(args.model_file_path)
        )
    model.load_weights(args.model_file_path)

    true_stages, predicted_stages = predict_stages(model, test_data)

    f1 = f1_score(true_stages, predicted_stages, average='macro')
    accuracy = accuracy_score(true_stages, predicted_stages)

    print(f'f1 score: {f1}')
    print(f'Accuracy score: {accuracy}')
    print(f'Classification report:\n{classification_report(true_stages, predicted_stages)}')


if __name__ == '__main__':
    main()
