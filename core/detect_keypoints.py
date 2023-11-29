from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt, xywh2xyxy
from torchvision import transforms
from utils.plots import output_to_keypoint

import numpy as np
import torch

def detect_keypoints(image, model,device):
    model.half().to(device)
    image_height, image_width, _ = image.shape
    image = letterbox(image)[0]
    nimage_height, nimage_width, _ = image.shape
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))

    image = image.half().to(device)
    output = model(image)[0]
    output = non_max_suppression_kpt(
        output, 0.25, 0.45, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
    with torch.no_grad():
        output = output_to_keypoint(output)
    kpt = [[] for _ in range(len(output))]
    bbox = []
    confidences = [[] for _ in range(len(output))]
    for i in range(len(output)):
        box = (output[i][2] / nimage_width * image_width, output[i][3] / nimage_height * image_height,
               output[i][4] / nimage_width * image_width, output[i][5] / nimage_height * image_height)
        bbox.append(box)
        kpts = output[i][7:]
        num_kpts = len(kpts) // 3
        for kid in range(num_kpts):
            x, y = kpts[3 * kid] / nimage_width * \
                image_width, kpts[3 * kid + 1] / nimage_height * image_height
            confidence = kpts[3 * kid + 2]
            if confidence > 0:
                kpt[i].append((int(x), int(y)))
                confidences[i].append(float(confidence))
    return [kpt, torch.Tensor(bbox) if len(bbox) == 0 else xywh2xyxy(torch.Tensor(bbox)), confidences]
