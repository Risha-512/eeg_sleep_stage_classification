import numpy as np
import pandas as pd

from os import path
from mne.io import read_raw_edf
from dataclasses import dataclass

from argparse import ArgumentParser

from edf_readers import EDFHeaderReader, SleepStageEDFReader, EDFData
from common.file_utils import *
from common.edf_parameters import *
from common.npz_parameters import *


def parse_arguments():
    """
    Получить входные аргументы программы:
        - (--input_directory) путь к директории с исходными данными
        - (--output_directory) путь к директории для записи новых файлов
    """
    parser = ArgumentParser()
    parser.add_argument('--input_directory', type=str, default=EDF_DIRECTORY_PATH, help='Путь к файлам edf')
    parser.add_argument('--output_directory', type=str, default=NPZ_DIRECTORY_PATH, help='Путь директории для записи')
    return parser.parse_args()


def read_edf_header(file_path: str) -> dict:
    """
    Считать заголовок edf файла

    :param file_path: путь к edf файлу
    :return: данные заголовка файла
    """
    with open(file_path, 'r', encoding=EDF_ENCODING) as file:
        return EDFHeaderReader(file).read_header()


def read_sleep_stages_from_edf(file_path: str) -> EDFData:
    """
    Считать данные (заголовок и записи стадий) стадий сна с edf файла

    :param file_path: путь к edf файлу
    :return: данные заголовка файла, записи стадий
    """
    with open(file_path, 'r', encoding=EDF_ENCODING) as file:
        return SleepStageEDFReader(file).read_header_and_records()


@dataclass
class RawStageData:
    raw_values: np.array
    stage_values: np.array


def select_signed_data(raw_data: pd.DataFrame, stages_data: pd.DataFrame, sampling_rate: int) -> RawStageData:
    """
    Выбрать данные, у которых известны соответствующие им стадии

    :param raw_data: данные ЭЭГ
    :param stages_data: данные соответствующих стадий
    :param sampling_rate: частоты выборки (частоты дискретизации ЭЭГ)
    :return: отфильтрованный массив показаний ЭЭГ, соответствующий массив стадий
    """
    stages_values, indices = np.array([], dtype=int), np.array([], dtype=int)

    for stage_data in stages_data.itertuples(index=False):
        # пропустить данные с неизвестной стадией
        if STAGE_ANNOTATIONS[stage_data.annotation] == UNKNOWN:
            continue

        # найти длительность эпохи и добавить стадии
        epoch_duration = int(stage_data.duration / EPOCH_SIZE)
        stages_values = np.append(stages_values,
                                  np.ones(epoch_duration, dtype=int) * STAGE_ANNOTATIONS[stage_data.annotation])

        # добавить индексы данных текущей итерации
        indices = np.append(indices, stage_data.onset * sampling_rate + np.arange(stage_data.duration * sampling_rate))

    # оставить только данные с известными стадиями
    return RawStageData(raw_data.values[indices], stages_values)


def remove_excess_stage_w_values(raw_values: np.array, stages_values: np.array, epochs_number: int) -> RawStageData:
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

    return RawStageData(raw_values[indices], stages_values[indices])


def save_data_to_npz(data_to_save: dict, output_directory: str, filename: str):
    """
    Сохранить данные в файл формата npz (numpy файл)

    :param data_to_save: данные для записи в файл
    :param output_directory: директория, в которую будут сохраняться файлы
    :param filename: имя исходного файла
    """
    np.savez(path.join(output_directory, filename + NPZ_FILE_EXTENSION), **data_to_save)


def main():
    args = parse_arguments()
    create_directory(args.output_directory)

    # получить пути всех edf файлов
    psg_files = get_files_in_directory(args.input_directory, PSG_FILE_PATTERN)
    hyp_files = get_files_in_directory(args.input_directory, HYPNOGRAM_FILE_PATTERN)

    for psg_file, hyp_file in zip(psg_files, hyp_files):
        # считать данные ЭЭГ
        raw_data = read_raw_edf(psg_file, preload=True, stim_channel=None)
        sampling_rate = int(raw_data.info[SAMPLING_RATE_INFO_KEY])
        raw_records = raw_data.to_data_frame()[CHANNEL_NAME].to_frame()

        # считать заголовки к данным ЭЭГ и данным стадий
        header_raw = read_edf_header(psg_file)
        stage_data = read_sleep_stages_from_edf(hyp_file)

        # выбрать данные, имеющие известные стадии
        data = select_signed_data(raw_records, stage_data.records, sampling_rate)

        # вычислить количество эпох
        epochs_number = int(len(data.raw_values) / (EPOCH_SIZE * sampling_rate))

        # выделить эпохи и их стадии
        data.raw_values = np.asarray(np.split(data.raw_values, epochs_number)).astype(np.float32)

        # проверить, что количество данных эквивалентно
        assert len(data.raw_values) == len(data.stage_values) == epochs_number

        # удалить лишние данные в стадии бодрствования
        data = remove_excess_stage_w_values(data.raw_values, data.stage_values, epochs_number)

        # сохранить данные в файл формата npz
        save_data_to_npz(
            dict(zip(NPZ_KEYS, [
                data.raw_values,
                data.stage_values,
                sampling_rate,
                CHANNEL_NAME,
                header_raw,
                stage_data.header
            ])),
            args.output_directory,
            get_file_name_from_path(psg_file)
        )


if __name__ == '__main__':
    main()
