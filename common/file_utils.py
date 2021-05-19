from os import path
from glob import glob
from typing import List


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
