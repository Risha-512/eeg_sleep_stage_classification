import datetime
import re

import numpy as np
import pandas as pd

from typing import List, Callable

HEADER_START = '0       '

EDF_PLUS_C = 'EDF+C'
EDF_PLUS_D = 'EDF+D'

EVENT_CHANNEL = 'EDF Annotations'


class EDFHeaderReader:
    def __init__(self, file):
        self.file = file

    def __read_list(self, item_bytes: int, size: int, function: Callable) -> list:
        """
        Считать список данных с файла

        :param item_bytes: размер элемента списка в байтах
        :param size: размер списка (количество элементов)
        :param function: функция обработки каждого элемента
        :return: список обработанных данных
        """
        return [function(self.file.read(item_bytes)) for n in range(size)]

    def __read_header(self) -> dict:
        """
        Считать данные заголовка файла

        :return: объект данных заголовка
        """
        # убедиться, что указатель стоит в начале файла
        assert self.file.tell() == 0
        assert self.file.read(8) == HEADER_START

        # данные человека и записи ЭЭГ
        header = {
            'local_subject_data': self.file.read(80).strip(),
            'local_recording_data': self.file.read(80).strip()
        }

        # выделить дату записи
        (day, month, year) = [int(x) for x in self.file.read(8).split('.')]
        (hour, minute, sec) = [int(x) for x in self.file.read(8).split('.')]

        year += 2000 if datetime.datetime.today().year - 2000 >= year else 1900
        header['date'] = str(datetime.datetime(year, month, day, hour, minute, sec))

        # размер заголовка
        header_bytes_num = int(self.file.read(8))

        # тип edf файла
        subtype = self.file.read(44)[:5]
        header['is_EDF+'] = subtype in [EDF_PLUS_C, EDF_PLUS_D]
        header['contiguous'] = subtype != EDF_PLUS_D

        # данные записей (показаний)
        header['records_num'] = int(self.file.read(8))
        header['record_length'] = float(self.file.read(8))
        header['channels_num'] = channels_num = int(self.file.read(4))

        header['channels'] = self.__read_list(item_bytes=16, size=channels_num, function=lambda x: x.strip())
        header['transducer'] = self.__read_list(item_bytes=80, size=channels_num, function=lambda x: x.strip())
        header['units'] = self.__read_list(item_bytes=8, size=channels_num, function=lambda x: x.strip())
        header['physical_min'] = self.__read_list(item_bytes=8, size=channels_num, function=lambda x: float(x))
        header['physical_max'] = self.__read_list(item_bytes=8, size=channels_num, function=lambda x: float(x))
        header['digital_min'] = self.__read_list(item_bytes=8, size=channels_num, function=lambda x: float(x))
        header['digital_max'] = self.__read_list(item_bytes=8, size=channels_num, function=lambda x: float(x))
        header['pre_filtration'] = self.__read_list(item_bytes=80, size=channels_num, function=lambda x: x.strip())
        header['samples_per_record'] = self.__read_list(item_bytes=8, size=channels_num, function=lambda x: int(x))

        # убедиться, что считаны все данные заголовка (учитывая, что должно остаться указанное количество записей)
        assert self.file.tell() == header_bytes_num - 32 * channels_num
        return header

    def read_header(self) -> dict:
        """
        Считать данные заголовка
        """
        header = self.__read_header()

        # убедиться, что данные корректны и максимальные значения не меньше минмальных
        assert np.all(np.asarray(header['physical_max']) - np.asarray(header['physical_min']) >= 0)
        assert np.all(np.asarray(header['digital_max']) - np.asarray(header['digital_min']) >= 0)

        return header


class SleepStageEDFReader:
    def __init__(self, file):
        self.__header_reader = EDFHeaderReader(file)

    @classmethod
    def __parse_tal_item(cls, stage: str) -> dict:
        stage_data = re.split('[\x15\x14]', stage)
        return {
            'onset': int(stage_data[0]),
            'duration': int(stage_data[1]) if stage_data[1] else 0,
            'annotation': stage_data[2]
        }

    @classmethod
    def __parse_tal(cls, tal_str: str) -> List[dict]:
        """
        Получить список кортежей (onset, duration, annotation) для EDF+ TAL
            - onset - момент начала стадии (сек)
            - duration - длительность стадии (сек)
            - annotation - аннотация к стадии (указывает тип стадии)

        Time-stamped Annotations Lists (TALs) - Списки аннотаций с отметкой времени

        :param tal_str: строка списка кортежей
        :return: список кортежей (onset, duration, annotation)
        """
        return [cls.__parse_tal_item(stage) for stage in tal_str.split('\x14\x00')[1:-1]]

    def __read_raw_records(self, samples_per_record: List[int]) -> List[str]:
        """
        Считать записи (данные стадий)

        :param samples_per_record: размеры выборкок в записях
        :return: список записей (данных стадий)
        """
        stages_records = []
        for num in samples_per_record:
            samples = self.__header_reader.file.read(num * 2)
            if len(samples) != num * 2:
                break
            stages_records.append(samples)
        return stages_records

    def __convert_stages_records(self, channels: List[str], stages_records: List[str]) -> pd.DataFrame:
        """
        Конвертировать записи в формат DataFrame (onset, duration, annotation),
        основываясь на данных заголовка.

        :param channels: записанные каналы ЭЭГ
        :param stages_records: записи файла (данные стадий)
        :return: DataFrame из кортежей (onset, duration, annotation)
        """
        events = []
        for (i, record) in enumerate(stages_records):
            if channels[i] == EVENT_CHANNEL:
                events.extend(self.__parse_tal(record))

        return pd.DataFrame(events)

    def read_header_and_records(self) -> (dict, pd.DataFrame):
        """
        Считать заголовок и записи стадий в виде DataFrame (onset, duration, annotation)

        :return: словарь данных заголовка, DataFrame из кортежей (onset, duration, annotation)
        """
        header = self.__header_reader.read_header()
        records = self.__convert_stages_records(header['channels'],
                                                self.__read_raw_records(header['samples_per_record']))
        return header, records
