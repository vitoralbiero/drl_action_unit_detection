from .file_cache_manager import FileCacheManager
import numpy as np
from os import makedirs, path


class CacheMultipleFiles(FileCacheManager):

    _directory_path = ''

    def __init__(self, directory_path):
        self._directory_path = directory_path

        if not path.exists(directory_path):
            makedirs(directory_path)

    def contains(self, key):
        key = self._file_path(key)
        return path.exists(key)

    def get(self, key):
        key = self._file_path(key)
        return np.load(key)

    def add(self, key, value):
        key = self._file_path(key)
        np.save(key, value, allow_pickle=False)

    def _file_path(self, key):
        file_name = key + '.npy'
        return path.join(self._directory_path, file_name)
