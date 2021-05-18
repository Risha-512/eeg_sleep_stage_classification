import numpy as np
import pandas as pd

from os import path, makedirs
from shutil import rmtree
from glob import glob
from argparse import ArgumentParser
from mne.io import read_raw_edf
from typing import List

from edf_readers import EDFHeaderReader, SleepStageEDFReader
from data_parameters import *


def parse_arguments():
    """
    Получить входные аргументы программы:
        - путь к директории с иходными данными
        - путь к директории для записи новых файлов
    """
    parser = ArgumentParser()
    parser.add_argument('--input_directory', type=str, default=INPUT_DIRECTORY_PATH, help='Directory of the edf files')
    parser.add_argument('--output_directory', type=str, default=OUTPUT_DIRECTORY_PATH, help='Directory for output')
    return parser.parse_args()


def get_files_in_directory(directory_path: str, file_pattern: str) -> List[str]:
    """
    Получить отсортированный список всех файлов в директории, удовлетворяющих паттерну

    :param directory_path: директория, из которой выбираются файлы
    :param file_pattern: паттерн имени файла
    :return: отсортированный список путей всех файлов
    """
    return sorted(glob(path.join(directory_path, file_pattern)))


def read_edf_header(file_path: str) -> dict:
    """
    Считать заголовок edf файла

    :param file_path: путь к edf файлу
    :return: данные заголовка файла
    """
    with open(file_path, 'r', encoding=ENCODING) as file:
        return EDFHeaderReader(file).read_header()


def read_sleep_stages_from_edf(file_path: str) -> (dict, pd.DataFrame):
    """
    Считать данные (заголовок и записи стадий) стадий сна с edf файла

    :param file_path: путь к edf файлу
    :return: данные заголовка файла, записи стадий
    """
    with open(file_path, 'r', encoding=ENCODING) as file:
        return SleepStageEDFReader(file).read_header_and_records()


def select_signed_data(raw_data: pd.DataFrame, stages_data: pd.DataFrame, sampling_rate: int) -> (np.array, np.array):
    """
    Выбрать данные, у которых известны соответствующие им стадии

    :param raw_data: данные ЭЭГ
    :param stages_data: данные соответствующих стадий
    :param sampling_rate: частоты выборки (частоты дискретизации ЭЭГ)
    :return: отфильтрованный массив показаний ЭЭГ, соответствующий массив стадий
    """
    stages_values = indices = np.array([], dtype=int)

    for stage_data in stages_data.itertuples(index=False):
        # пропустить данные с неизвестной стадией
        if STAGES_TYPES[stage_data.annotation] == UNKNOWN:
            continue

        # найти длительность эпохи и добавить стадии
        epoch_duration = int(stage_data.duration / EPOCH_SIZE)
        stages_values = np.append(stages_values,
                                  np.ones(epoch_duration, dtype=int) * STAGES_TYPES[stage_data.annotation])

        # добавить индексы данных текущей итерации
        indices = np.append(indices, stage_data.onset * sampling_rate + np.arange(stage_data.duration * sampling_rate))

    # оставить только данные с известными стадиями
    return raw_data.values[indices], stages_values


def remove_excess_stage_w_values(raw_values: np.array, stages_values: np.array, epochs_number: int) -> (np.array, np.array):
    """
    Удалить излишние данные в стадии W (стадия бодрствования)
    
    :param raw_values: показания ЭЭГ
    :param stages_values: соответствующие стадии
    :param epochs_number: количество эпох
    :return: отфильтрованный массив показаний ЭЭГ, соответствующий массив стадий
    """
    # получить массив стадий без стадии W
    without_w_idx = np.where(stages_values != W)[0]

    # оставить по возможности четыре (по две от начала и с конца) эпохи стадии W
    start_index = max(0, without_w_idx[0] - EPOCH_SIZE * 2)
    end_index = min(epochs_number - 1, without_w_idx[-1] + EPOCH_SIZE * 2)

    indices = np.arange(start_index, end_index + 1)

    return raw_values[indices], stages_values[indices]


def save_data_to_npz(data_to_save: dict, output_directory: str, psg_filename: str):
    """
    Сохранить данные в файл формата npz (numpy файл)

    :param data_to_save: данные для записи в файл
    :param output_directory: директория, в которую будут сохраняться файлы
    :param psg_filename: имя исходного файла ПСГ (edf файл с показаниями ЭЭГ)
    """
    npz_filename = psg_filename.replace(PSG_FILE_EXTENSION, NPZ_FILE_EXTENSION)
    np.savez(path.join(output_directory, npz_filename), **data_to_save)


def main():
    args = parse_arguments()

    # создать директорию для выходных данных (очистить, если существует)
    if path.exists(args.output_directory):
        rmtree(args.output_directory)
    makedirs(args.output_directory)

    # получить пути всех edf файлов
    psg_files = get_files_in_directory(args.input_directory, PSG_FILE_PATTERN)
    hyp_files = get_files_in_directory(args.input_directory, HYPNOGRAM_FILE_PATTERN)

    for psg_file, hyp_file in zip(psg_files, hyp_files):
        # считать данные ЭЭГ
        raw_data = read_raw_edf(psg_file, preload=True, stim_channel=None)
        sampling_rate = int(raw_data.info[SAMPLING_RATE_INFO_KEY])

        raw_data = raw_data.to_data_frame()[CHANNEL_NAME].to_frame()

        # считать заголовки к данным ЭЭГ и данным стадий
        header_raw = read_edf_header(psg_file)
        header_stages, stages_data = read_sleep_stages_from_edf(hyp_file)

        # выбрать данные, имеющие известные стадии
        raw_values, stages_values = select_signed_data(raw_data, stages_data, sampling_rate)

        # вычислить количество эпох
        epochs_number = int(len(raw_values) / (EPOCH_SIZE * sampling_rate))

        # выделить эпохи и их стадии
        raw_values = np.asarray(np.split(raw_values, epochs_number)).astype(np.float32)

        # проверить, что количество данных эквивалентно
        assert len(raw_values) == len(stages_values) == epochs_number

        # удалить лишние данные в стадии бодрствования
        raw_values, stages_values = remove_excess_stage_w_values(raw_values, stages_values, epochs_number)

        # сохранить данные в файл формата npz
        save_data_to_npz(
            {
                'x': raw_values,
                'y': stages_values,
                'sampling_rate': sampling_rate,
                'channel_name': CHANNEL_NAME,
                "header_raw": header_raw,
                "header_stages": header_stages,
            },
            args.output_directory,
            path.basename(psg_file)
        )


if __name__ == '__main__':
    main()
