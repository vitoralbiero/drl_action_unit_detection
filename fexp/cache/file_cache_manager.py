import abc


class FileCacheManager(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def contains(self, key):
        msg = "contains(self, key) must be implemented."
        raise NotImplementedError(msg)

    @abc.abstractmethod
    def add(self, key, value):
        msg = "add(self, key, value) must be implemented."
        raise NotImplementedError(msg)

    @abc.abstractmethod
    def get(self, key):
        msg = "get(self, key) must be implemented."
        raise NotImplementedError(msg)
