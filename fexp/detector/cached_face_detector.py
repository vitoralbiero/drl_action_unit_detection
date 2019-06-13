import cv2
import logging
from os import path


class CachedFaceDetector(object):
    _face_detector = None
    _cache_manager = None

    def __init__(self, face_detector, cache_factory):
        self._face_detector = face_detector
        self._cache_manager = cache_factory.manager('face_bounding_box',
                                                    dtype=int)

    def detect_face(self, image, key):
        if self._cache_manager.contains(key):
            logging.info('Loading cached bounding box for %s.', key)
            return self._cache_manager.get(key)

        logging.info('Detecting the face for %s.', key)
        bbox = self._face_detector.detect_face(image)

        if bbox is not None:
            logging.info('Bounding box detected for %s, saving to cache.', key)
            self._cache_manager.add(key, bbox)

        return bbox
