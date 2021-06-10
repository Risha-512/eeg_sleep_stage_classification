from os import path, makedirs
from glob import glob
from typing import List, Sized

TXT_EXTENSION = '.txt'
PNG_EXTENSION = '.png'


def split_into_chunks(data: Sized, chunk_size: int) -> list:
    """
    Разделить данные на равные блоки

    :param data: данные, которые необходимо поделить
    :param chunk_size: размер блока
    :return: список из разделенных на блоки данных
    """
    return [data[idx:idx + chunk_size] for idx in range(0, len(data), chunk_size)]


def get_files_in_directory(directory_path: str, file_pattern: str) -> List[str]:
    """
    Получить отсортированный список всех файлов в директории, удовлетворяющих паттерну

    :param directory_path: директория, из которой выбираются файлы
    :param file_pattern: паттерн имени файла
    :return: отсортированный список путей всех файлов
    """
    return sorted(glob(path.join(directory_path, file_pattern)))


def get_file_name_from_path(file_path: str, with_extension: bool = False) -> str:
    """
    Получить имя файла из пути к нему

    :param file_path: путь к файлу
    :param with_extension: True, если нужно вернуть имя файла с расширением
    :return: имя файла
    """
    if with_extension:
        return path.basename(file_path)

    return path.basename(file_path).rsplit('.', 1)[0]


def create_directory(directory_path: str):
    """
    Создать директорию

    :param directory_path: путь директории
    """
    if not path.isdir(directory_path):
        makedirs(directory_path)


def write_to_text_file(file_path: str, data: List[str]):
    """
    Записать список строк в файл

    :param file_path: путь к текстовому файлу
    :param data: данные, которые необходимо записать
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(map(lambda x: x + '\n', data))
