from os import path, makedirs
from glob import glob
from typing import List, Sized

TXT_EXTENSION = '.txt'
PNG_EXTENSION = '.png'


def split_into_chunks(data: Sized, chunk_size: int) -> list:
    return [data[idx:idx + chunk_size] for idx in range(0, len(data), chunk_size)]


def get_files_in_directory(directory_path: str, file_pattern: str) -> List[str]:
    return sorted(glob(path.join(directory_path, file_pattern)))


def get_file_name_from_path(file_path: str, with_extension: bool = False) -> str:
    if with_extension:
        return path.basename(file_path)

    return path.basename(file_path).rsplit('.', 1)[0]


def create_directory(directory_path: str):
    if not path.isdir(directory_path):
        makedirs(directory_path)


def write_to_text_file(file_path: str, data: List[str]):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(map(lambda x: x + '\n', data))
