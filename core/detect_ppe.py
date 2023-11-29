from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords

import numpy as np
import torch


def detect_ppe(image, model,device):
    image_height, image_width, _ = image.shape

    new_image = letterbox(image)[0]
    # Convert the image to a tensor and perform object detection
    tensor_image = torch.from_numpy(np.ascontiguousarray(
        new_image[:, :, ::-1].transpose(2, 0, 1))).to(device)
    tensor_image = tensor_image.half()  # uint8 to fp16/32
    tensor_image /= 255.0  # 0 - 255 to 0.0 - 1.0
    tensor_image = tensor_image.unsqueeze(0)
    with torch.no_grad():
        out, _ = model(tensor_image)

    boxes = []
    confidences = []
    class_ids = []
    out = non_max_suppression(out, conf_thres=0.25, iou_thres=0.45)
    for detection in out[0]:
        confidence = detection[4]
        class_id = detection[5]
        x1 = detection[0]
        y1 = detection[1]
        x2 = detection[2]
        y2 = detection[3]
        x1, y1, x2, y2 = scale_coords([new_image.shape[0], new_image.shape[1]], torch.Tensor(
            [[x1, y1, x2, y2]]), [image_height, image_width])[0]
        boxes.append((int(x1), int(y1), int(x2), int(y2)))
        confidences.append(float(confidence))
        class_ids.append(int(class_id))

    return [boxes, confidences, class_ids]
