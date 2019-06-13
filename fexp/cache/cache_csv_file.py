from .file_cache_manager import FileCacheManager
import numpy as np
from os import path


class CacheCsvFile(FileCacheManager):

    _cache_file = ''
    _cache = None
    _dtype = None

    def __init__(self, cache_file, dtype=np.float64):
        self._cache_file = cache_file
        self._dtype = dtype
        self._load_cache(cache_file)

    def _load_cache(self, cache_file):
        self._cache = {}

        if not path.exists(cache_file):
            return

        with open(cache_file, 'r+') as file:
            for line in file:
                line = line.split(',')
                key = line[0]
                values = [self._dtype(v) for v in line[1:]]
                self._cache[key] = np.asarray(values)

    def contains(self, key):
        return key in self._cache

    def get(self, key):
        return self._cache[key]

    def add(self, key, value):
        self._cache[key] = value
        self._add_content_to_file(key, value)

    def _add_content_to_file(self, key, value):
        values = [repr(v) for v in value]
        values = ','.join(values)
        cache_row = '{},{}\n'.format(key, values)

        with open(self._cache_file, 'a+') as file:
            file.write(cache_row)
