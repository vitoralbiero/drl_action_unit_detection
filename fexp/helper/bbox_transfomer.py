from __future__ import division
import cv2
from fexp.helper import rectangle
import numpy as np


def _resize(object_detected, object_size, stretch=False):
    if object_size == 0 and not stretch:
        return object_detected

    x_size = object_detected.shape[1]
    y_size = object_detected.shape[0]

    bigger_side = max(x_size, y_size)
    smaller_side = min(x_size, y_size)

    if int(smaller_side == 0):
        smaller_side = 1

    aspect_ratio = float(bigger_side) / float(smaller_side)

    if object_size != 0:
        if x_size > y_size:
            x_size = object_size
            y_size = object_size / aspect_ratio
        else:
            y_size = object_size
            x_size = object_size / aspect_ratio

    if stretch:
        x_size = max(x_size, y_size)
        y_size = x_size

    x_size = int(np.rint(x_size))
    y_size = int(np.rint(y_size))

    return cv2.resize(object_detected, (x_size, y_size), cv2.INTER_CUBIC)


def _shape_divisible_by_four(image):
    (height, width) = image.shape[:2]
    height = round(height / 4) * 4
    width = round(width / 4) * 4

    return cv2.resize(image, (width, height),
                      interpolation=cv2.INTER_CUBIC)


def special_bbox(bbox, image, object_size):
    (left, top, width, height) = rectangle.bbox_auaanet(bbox, object_size)
    right = width + left
    bottom = height + top

    object_detected = image[top:bottom, left:right]
    return _resize(object_detected, object_size)


def original_bbox_div_by_four(bbox, image, object_size):
    left = bbox[0]
    top = bbox[1]
    right = bbox[2] + left
    bottom = bbox[3] + top

    object_detected = image[top:bottom, left:right]
    image = _resize(object_detected, object_size)

    return _shape_divisible_by_four(image)


def original_bbox(bbox, image, object_size):
    left = bbox[0]
    top = bbox[1]
    right = bbox[2] + left
    bottom = bbox[3] + top

    object_detected = image[top:bottom, left:right]
    return _resize(object_detected, object_size)


def stretched_bbox(bbox, image, object_size):
    left = bbox[0]
    top = bbox[1]
    right = bbox[2] + left
    bottom = bbox[3] + top

    object_detected = image[top:bottom, left:right]
    return _resize(object_detected, object_size, True)


def squared_bbox(bbox, image, object_size):
    (left, top, size, size) = rectangle.bbox_to_square(bbox)
    right = size + left
    bottom = size + top

    object_detected = image[top:bottom, left:right]
    return _resize(object_detected, object_size)


def squared_center_bbox(bbox, image, object_size):
    (left, top, size, size) = rectangle.bbox_to_square_center(bbox)
    right = size + left
    bottom = size + top

    object_detected = image[top:bottom, left:right]
    return _resize(object_detected, object_size)
