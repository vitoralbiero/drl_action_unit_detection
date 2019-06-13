from .file_cache_manager import FileCacheManager


class NoCacheFile(FileCacheManager):

    def contains(self, key):
        return False

    def add(self, key, value):
        pass

    def get(self, key):
        raise Exception('get() should not be called if not contains key.')
