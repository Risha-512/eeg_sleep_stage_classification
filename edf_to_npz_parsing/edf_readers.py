import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from typing import List, Callable, TextIO
from datetime import datetime

from edf_header_keys import *


class EDFHeaderReader:
    def __init__(self, file: TextIO):
        self._file = file
        self.__header_start = '0       '
        self.__edf_plus_c = 'EDF+C'
        self.__edf_plus_d = 'EDF+D'

    def __read_list(self, item_bytes: int, size: int, function: Callable) -> list:
        return [function(self._file.read(item_bytes)) for n in range(size)]

    def __read_header(self) -> dict:
        # make sure the pointer is at the beginning of the file
        assert self._file.tell() == 0
        assert self._file.read(8) == self.__header_start

        header = {
            LOCAL_SUBJECT_DATA_KEY: self._file.read(80).strip(),
            LOCAL_RECORDING_DATA_KEY: self._file.read(80).strip()
        }

        # date of the recording
        (day, month, year) = [int(x) for x in self._file.read(8).split('.')]
        (hour, minute, sec) = [int(x) for x in self._file.read(8).split('.')]

        year += 2000 if datetime.today().year - 2000 >= year else 1900
        header[DATE_KEY] = str(datetime(year, month, day, hour, minute, sec))

        header_bytes_num = int(self._file.read(8))

        # edf type
        subtype = self._file.read(44)[:5]
        header[IS_EDF_PLUS_KEY] = subtype in [self.__edf_plus_c, self.__edf_plus_d]
        header[IS_CONTIGUOUS_KEY] = subtype != self.__edf_plus_d

        # recordings
        header[RECORDS_NUM_KEY] = int(self._file.read(8))
        header[RECORD_DURATION] = float(self._file.read(8))
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

        # make sure all data has been read
        assert self._file.tell() == header_bytes_num - 32 * channels_num
        return header

    def read_header(self) -> dict:
        header = self.__read_header()

        # make sure max is greater than min
        assert np.all(np.asarray(header[PHYSICAL_MAX_KEY]) - np.asarray(header[PHYSICAL_MIN_KEY]) >= 0)
        assert np.all(np.asarray(header[DIGITAL_MAX_KEY]) - np.asarray(header[DIGITAL_MIN_KEY]) >= 0)

        return header


@dataclass
class EDFData:
    header: dict
    records: pd.DataFrame


class SleepStageEDFReader(EDFHeaderReader):
    def __init__(self, file: TextIO):
        super().__init__(file)
        self.__event_channel = 'EDF Annotations'

    @classmethod
    def __parse_tal_item(cls, stage: str) -> dict:
        """
        Get TAL tuple: (onset, duration, annotation)

        :param stage: stage data
        :return: tuple (onset, duration, annotation)
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
        Get list of tuples (onset, duration, annotation) for EDF+ TAL
            - onset - stage beginning (sec)
            - duration - stage duration (sec)
            - annotation - stage annotation (stage type)

        Time-stamped Annotations Lists (TALs)

        :param tal_str: string of tuple list
        :return: list of tuples (onset, duration, annotation)
        """
        return [cls.__parse_tal_item(stage) for stage in tal_str.split('\x14\x00')[1:-1]]

    def __read_raw_records(self, samples_per_record: List[int]) -> List[str]:
        stages_records = []
        for num in samples_per_record:
            samples = self._file.read(num * 2)
            if len(samples) != num * 2:
                break
            stages_records.append(samples)
        return stages_records

    def __convert_stages_records(self, channels: List[str], stages_records: List[str]) -> pd.DataFrame:
        events = []
        for (i, record) in enumerate(stages_records):
            if channels[i] == self.__event_channel:
                events.extend(self.__parse_tal(record))

        return pd.DataFrame(events)

    def read_header_and_records(self) -> EDFData:
        header = self.read_header()
        records = self.__convert_stages_records(header[CHANNELS_KEY],
                                                self.__read_raw_records(header[SAMPLES_PER_RECORD_KEY]))
        return EDFData(header, records)
