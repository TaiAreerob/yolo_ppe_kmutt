import math
import torch
import numpy as np


def is_inside(outer_box, inner_box, threshold):
    x1, y1, x2, y2 = outer_box
    x3, y3, x4, y4 = inner_box

    intersection_area = max(0, min(x2, x4) - max(x1, x3)) * \
        max(0, min(y2, y4) - max(y1, y3))

    inner_box_area = (x4 - x3) * (y4 - y3)

    IoI = intersection_area / inner_box_area

    return IoI >= threshold


def are_near(bounding_box, keypoint, threshold_percentage):
    x1, y1, x2, y2 = bounding_box

    width = x2 - x1
    height = y2 - y1

    size = math.sqrt(width**2 + height**2)

    threshold = threshold_percentage * size

    keypoint_x, keypoint_y = keypoint

    center_x = x1 + (width / 2)
    center_y = y1 + (height / 2)

    distance = math.sqrt((keypoint_x - center_x)**2 +
                         (keypoint_y - center_y)**2)

    return distance < threshold


def middle_point(points):
    n = len(points)

    sum_x = 0
    sum_y = 0

    for point in points:
        sum_x += point[0]
        sum_y += point[1]

    avg_x = sum_x / n
    avg_y = sum_y / n

    return (avg_x, avg_y)


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x1 >= x2 or y1 >= y2:
        return 0
    intersection = (x2 - x1) * (y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    iou = intersection / union
    return iou


def xyxy2xywh_topleft(x):
    if x.ndimension() != 2:
        return x
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y
