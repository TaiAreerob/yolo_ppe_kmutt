from general.general import xyxy2xywh_topleft
from deepSort.detection import Detection

import numpy as np
import torch


def apply_track(tracker, encoder, image, boxes, confidences, class_ids):
    boxes = xyxy2xywh_topleft(torch.Tensor(np.array(boxes)))
    features = np.array(encoder(image, boxes))
    detections = [Detection(bbox, score, class_ids, feature) for bbox, score,
                  class_ids, feature in zip(boxes, confidences, class_ids, features)]
    tracker.predict()
    tracker.update(detections)
    boxes, confidences, class_ids, track_ids = [[], [], [], []]
    for track in tracker.tracks:
        boxes.append(track.to_tlbr())
        track_ids.append(track.track_id)
        class_ids.append(int(track.get_class()))
        confidences.append(int(track.get_class()))

    return boxes, confidences, class_ids, track_ids
