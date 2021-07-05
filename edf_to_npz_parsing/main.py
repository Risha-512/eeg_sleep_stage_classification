import numpy as np
import pandas as pd

from mne.io import read_raw_edf
from dataclasses import dataclass

from argparse import ArgumentParser

from edf_readers import EDFHeaderReader, SleepStageEDFReader, EDFData
from common.utils import *
from common.edf_parameters import *
from common.npz_parameters import *


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--input_directory', type=str, default=EDF_DIRECTORY_PATH, help='Path to EDF files')
    parser.add_argument('--output_directory', type=str, default=NPZ_DIRECTORY_PATH, help='Path to npz files')
    return parser.parse_args()


def read_edf_header(file_path: str) -> dict:
    with open(file_path, 'r', encoding=EDF_ENCODING) as file:
        return EDFHeaderReader(file).read_header()


def read_sleep_stages_from_edf(file_path: str) -> EDFData:
    with open(file_path, 'r', encoding=EDF_ENCODING) as file:
        return SleepStageEDFReader(file).read_header_and_records()


@dataclass
class RawStageData:
    raw_values: np.array
    stage_values: np.array


def select_signed_data(raw_data: pd.DataFrame, stages_data: pd.DataFrame, sampling_rate: int) -> RawStageData:
    stages_values, indices = np.array([], dtype=int), np.array([], dtype=int)

    for stage_data in stages_data.itertuples(index=False):
        # skip data with unknown stage
        if STAGE_ANNOTATIONS[stage_data.annotation] == UNKNOWN:
            continue

        # calculate epoch duration and append stages
        epoch_duration = int(stage_data.duration / EPOCH_SIZE)
        stages_values = np.append(stages_values,
                                  np.ones(epoch_duration, dtype=int) * STAGE_ANNOTATIONS[stage_data.annotation])

        # append indices of current iteration data
        indices = np.append(indices, stage_data.onset * sampling_rate + np.arange(stage_data.duration * sampling_rate))

    # keep only signed data
    return RawStageData(raw_data.values[indices], stages_values)


def remove_excess_stage_w_values(raw_values: np.array, stages_values: np.array, epochs_number: int) -> RawStageData:
    # get array without W stage
    without_w_idx = np.where(stages_values != W)[0]

    # keep 4 epochs of W stage (if possible): 2 before record beginning and 2 after record ending
    start_index = max(0, without_w_idx[0] - EPOCH_SIZE * 2)
    end_index = min(epochs_number - 1, without_w_idx[-1] + EPOCH_SIZE * 2)

    indices = np.arange(start_index, end_index + 1)

    return RawStageData(raw_values[indices], stages_values[indices])


def save_data_to_npz(data_to_save: dict, output_directory: str, filename: str):
    np.savez(path.join(output_directory, filename + NPZ_FILE_EXTENSION), **data_to_save)


def main():
    args = parse_arguments()
    create_directory(args.output_directory)

    # get all edf file paths
    psg_files = get_files_in_directory(args.input_directory, PSG_FILE_PATTERN)
    hyp_files = get_files_in_directory(args.input_directory, HYPNOGRAM_FILE_PATTERN)

    for psg_file, hyp_file in zip(psg_files, hyp_files):
        # read eeg data
        raw_data = read_raw_edf(psg_file, preload=True, stim_channel=None)
        sampling_rate = int(raw_data.info[SAMPLING_RATE_INFO_KEY])
        raw_records = raw_data.to_data_frame()[CHANNEL_NAME].to_frame()

        # read header raw and stage data
        header_raw = read_edf_header(psg_file)
        stage_data = read_sleep_stages_from_edf(hyp_file)

        # select data
        data = select_signed_data(raw_records, stage_data.records, sampling_rate)

        # calculate epochs number
        epochs_number = int(len(data.raw_values) / (EPOCH_SIZE * sampling_rate))

        # split data into epochs
        data.raw_values = np.asarray(np.split(data.raw_values, epochs_number)).astype(np.float32)

        # make sure the data length is equivalent
        assert len(data.raw_values) == len(data.stage_values) == epochs_number

        # remove excess stages
        data = remove_excess_stage_w_values(data.raw_values, data.stage_values, epochs_number)

        # save data
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
