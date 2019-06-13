from __future__ import division
import numpy as np


def faster_rcnn_to_opencv(rectangle):
    left = rectangle[0]
    top = rectangle[1]
    width = rectangle[2] - left
    height = rectangle[3] - top

    return (left, top, width, height)


def bbox_to_square(bbox):
    left = bbox[0]
    top = bbox[1]
    width = bbox[2]
    height = bbox[3]

    size = max(width, height)

    left -= max(left + size, 1024) - 1024
    top -= max(top + size, 1024) - 1024

    return (left, top, size, size)


def bbox_to_square_center(bbox):
    left = bbox[0]
    top = bbox[1]
    width = bbox[2]
    height = bbox[3]

    size = max(width, height)
    diff = abs(height - width)

    if width > height:
        top -= int(diff / 2)
        top = max(0, top)

    if height > width:
        left -= int(diff / 2)
        left = max(0, left)

    left -= max(left + size, 1024) - 1024
    top -= max(top + size, 1024) - 1024

    return (left, top, size, size)


def bbox_auaanet(bbox, object_size):
    y_range = [120, 100, 80]
    x_range = [120, 100, 80, 60]

    left = bbox[0]
    top = bbox[1]
    width = bbox[2]
    height = bbox[3]

    x_size = width
    y_size = height

    bigger_side = max(x_size, y_size)
    smaller_side = min(x_size, y_size)

    aspect_ratio = float(bigger_side) / float(smaller_side)

    if object_size != 0:
        if x_size > y_size:
            x_size = object_size
            y_size = object_size / aspect_ratio
            y_size = min(y_range, key=lambda x: abs(x - y_size))

            new_aspect_ratio = x_size / y_size
            height = int(np.rint(width / new_aspect_ratio))
            top -= int(np.rint((height - bbox[3]) / 2))
            top = max(0, top)
        else:
            y_size = object_size
            x_size = object_size / aspect_ratio
            x_size = min(x_range, key=lambda x: abs(x - x_size))

            new_aspect_ratio = y_size / x_size
            width = int(np.rint(height / new_aspect_ratio))
            left -= int(np.rint((width - bbox[2]) / 2))
            left = max(0, left)

    return (left, top, width, height)
