import re

import numpy as np
import pandas as pd

from typing import List, Callable
from datetime import datetime

from edf_header_keys import *


class EDFHeaderReader:
    def __init__(self, file):
        self._file = file
        self.__header_start = '0       '
        self.__edf_plus_c = 'EDF+C'
        self.__edf_plus_d = 'EDF+D'

    def __read_list(self, item_bytes: int, size: int, function: Callable) -> list:
        """
        Считать список данных с файла

        :param item_bytes: размер элемента списка в байтах
        :param size: размер списка (количество элементов)
        :param function: функция обработки каждого элемента
        :return: список обработанных данных
        """
        return [function(self._file.read(item_bytes)) for n in range(size)]

    def __read_header(self) -> dict:
        """
        Считать данные заголовка файла

        :return: объект данных заголовка
        """
        # убедиться, что указатель стоит в начале файла
        assert self._file.tell() == 0
        assert self._file.read(8) == self.__header_start

        # данные человека и записи ЭЭГ
        header = {
            LOCAL_SUBJECT_DATA_KEY: self._file.read(80).strip(),
            LOCAL_RECORDING_DATA_KEY: self._file.read(80).strip()
        }

        # выделить дату записи
        (day, month, year) = [int(x) for x in self._file.read(8).split('.')]
        (hour, minute, sec) = [int(x) for x in self._file.read(8).split('.')]

        year += 2000 if datetime.today().year - 2000 >= year else 1900
        header[DATE_KEY] = str(datetime(year, month, day, hour, minute, sec))

        # размер заголовка
        header_bytes_num = int(self._file.read(8))

        # тип edf файла
        subtype = self._file.read(44)[:5]
        header[IS_EDF_PLUS_KEY] = subtype in [self.__edf_plus_c, self.__edf_plus_d]
        header[IS_CONTIGUOUS_KEY] = subtype != self.__edf_plus_d

        # данные записей (показаний)
        header[RECORDS_NUM_KEY] = int(self._file.read(8))
        header[RECORD_LENGTH_KEY] = float(self._file.read(8))
        header[CHANNELS_NUM_KEY] = channels_num = int(self._file.read(4))

        header[CHANNELS_KEY] = self.__read_list(item_bytes=16, size=channels_num, function=lambda x: x.strip())
        header[TRANSDUCER_KEY] = self.__read_list(item_bytes=80, size=channels_num, function=lambda x: x.strip())
        header[UNITS_KEY] = self.__read_list(item_bytes=8, size=channels_num, function=lambda x: x.strip())
        header[PHYSICAL_MIN_KEY] = self.__read_list(item_bytes=8, size=channels_num, function=lambda x: float(x))
        header[PHYSICAL_MAX_KEY] = self.__read_list(item_bytes=8, size=channels_num, function=lambda x: float(x))
        header[DIGITAL_MIN_KEY] = self.__read_list(item_bytes=8, size=channels_num, function=lambda x: float(x))
        header[DIGITAL_MAX_KEY] = self.__read_list(item_bytes=8, size=channels_num, function=lambda x: float(x))
        header[PRE_FILTRATION_KEY] = self.__read_list(item_bytes=80, size=channels_num, function=lambda x: x.strip())
        header[SAMPLES_PER_RECORD_KEY] = self.__read_list(item_bytes=8, size=channels_num, function=lambda x: int(x))

        # убедиться, что считаны все данные заголовка (учитывая, что должно остаться указанное количество записей)
        assert self._file.tell() == header_bytes_num - 32 * channels_num
        return header

    def read_header(self) -> dict:
        """
        Считать данные заголовка
        """
        header = self.__read_header()

        # убедиться, что данные корректны и максимальные значения не меньше минмальных
        assert np.all(np.asarray(header[PHYSICAL_MAX_KEY]) - np.asarray(header[PHYSICAL_MIN_KEY]) >= 0)
        assert np.all(np.asarray(header[DIGITAL_MAX_KEY]) - np.asarray(header[DIGITAL_MIN_KEY]) >= 0)

        return header


class SleepStageEDFReader(EDFHeaderReader):
    def __init__(self, file):
        super().__init__(file)
        self.__event_channel = 'EDF Annotations'

    @classmethod
    def __parse_tal_item(cls, stage: str) -> dict:
        """
        Получить кортеж (onset, duration, annotation)

        :param stage: строка данных стадии
        :return: кортеж (onset, duration, annotation)
        """
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
            samples = self._file.read(num * 2)
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
            if channels[i] == self.__event_channel:
                events.extend(self.__parse_tal(record))

        return pd.DataFrame(events)

    def read_header_and_records(self) -> (dict, pd.DataFrame):
        """
        Считать заголовок и записи стадий в виде DataFrame (onset, duration, annotation)

        :return: словарь данных заголовка, DataFrame из кортежей (onset, duration, annotation)
        """
        header = self.read_header()
        records = self.__convert_stages_records(header[CHANNELS_KEY],
                                                self.__read_raw_records(header[SAMPLES_PER_RECORD_KEY]))
        return header, records
