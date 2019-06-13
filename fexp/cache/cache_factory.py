from os import path
from .cache_csv_file import CacheCsvFile
from .cache_multiple_files import CacheMultipleFiles
from .no_cache_file import NoCacheFile
import numpy as np


class CacheFactory(object):

    _cache_directory = ''
    _no_cache = False

    def __init__(self, cache_directory, no_cache):
        self._cache_directory = cache_directory
        self._no_cache = no_cache

    def multiple_files(self, directory_name):
        directory_path = path.join(self._cache_directory, directory_name)

        if self._no_cache:
            return NoCacheFile()

        return CacheMultipleFiles(directory_path)

    def manager(self, file_name, dtype=np.float64):
        file_name = file_name + '.csv'
        file_path = path.join(self._cache_directory, file_name)

        if self._no_cache:
            return NoCacheFile()

        return CacheCsvFile(file_path, dtype=dtype)
