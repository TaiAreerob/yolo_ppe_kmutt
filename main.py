from core.detect_keypoints import detect_keypoints
from core.detect_ppe import detect_ppe
from general.false_positive_removal import filter_ppe, get_bounding_box_inside_person, get_bounding_box_nearby_bodypart
from general.plots import draw_bounding_box
from general.post_process import assign_ppe_to_person
from general.tracking import apply_track
from numpy import random

import cv2
import numpy as np
import time
import numpy as np
import torch
import argparse
import pathlib
import os 
from deepSort import nn_matching
from deepSort.tracker import Tracker
from deepSort import generate_detections as gdet

from models.yolo import Model
from models.pose_yolo import Model as PoseModel

from utils.torch_utils import select_device
def load_input(input_path):
    # Load the input image or video
    if input_path.endswith('.mp4'):
        video = cv2.VideoCapture(input_path)
        success, image = video.read()
    else:
        image = cv2.imread(input_path)
        success = True
        video = None

    return success, image, video

def get_model(path_or_model='path/to/model.pt',device='0' , autoshape=True, isPose=False):
    model = torch.load(path_or_model) if isinstance(path_or_model, str) else path_or_model  # load checkpoint
    if isinstance(model, dict):
        model = model['ema' if model.get('ema') else 'model']  # load model

    hub_model = Model(model.yaml).to(next(model.parameters()).device) if not isPose else PoseModel(model.yaml).to(next(model.parameters()).device)  # create
    hub_model.load_state_dict(model.float().state_dict())  # load state_dict
    hub_model.names = model.names  # class names
    if autoshape:
        hub_model = hub_model.autoshape()  # for file/URI/PIL/cv2/np inputs and NMS
    # device = select_device('0' if torch.cuda.is_available() else 'cpu')  # default to GPU if available
    return hub_model.to(device)

def detect_and_draw(input_path, output_path, device):
    head, tail = os.path.split("/tmp/d/a.dat")
    detection_path = str(pathlib.Path().resolve())+'/weights/best.pt'
    kpt_path = str(pathlib.Path().resolve())+'/weights/yolov7-w6-pose.pt'

    model = get_model(detection_path,device=device).to(device).eval()
    kpt_model = get_model(kpt_path,isPose=True,device=device).to(device).eval()
    
    labels = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in labels]
    
    max_cosine_distance = 0.2
    nn_budget = None
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, max_iou_distance=0.5, max_age=5)
    model_filename = './deepSort/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    success, image, video = load_input(input_path)
    
    image_height, image_width, _ = image.shape

    frame_rate = 30.0
    f = open(output_path+'/output.txt', 'w')
    total_time = 0
    person_ids = []
    print("processing...")
    while success:
        start_time = time.time()

        kpts, person_bbox, kpts_confidences = detect_keypoints(
            image, kpt_model,device)
        print(kpts_confidences)    
        boxes, confidences, class_ids = detect_ppe(image, model,device)

        boxes,confidences,class_ids = get_bounding_box_inside_person(boxes,person_bbox,confidences,class_ids,[0.5,0.5,0.5,0.5])
        boxes,confidences,class_ids = get_bounding_box_nearby_bodypart(kpts,boxes,confidences,class_ids,[1,3,1,0.5])
        boxes,confidences,class_ids = filter_ppe(person_bbox,boxes,confidences,class_ids,[2,1,2,1])
        
        boxes, confidences, class_ids, track_ids = apply_track(tracker, encoder, image, [*boxes, *person_bbox.numpy(
        )], [*confidences, *[np.mean(x) for x in kpts_confidences]], [*class_ids, *[5 for x in person_bbox]])
        ppe_boxes, ppe_confidences, ppe_class_ids, ppe_track_ids = [
            [], [], [], []]
        person_bbox, person_confidences, person_class_ids, person_track_ids = [
            [], [], [], []]
        for i in range(len(boxes)):
            if class_ids[i] == 5:
                person_bbox.append(boxes[i])
                person_confidences.append(confidences[i])
                person_class_ids.append(class_ids[i])
                person_track_ids.append(track_ids[i])
            else:
                ppe_boxes.append(boxes[i])
                ppe_confidences.append(confidences[i])
                ppe_class_ids.append(class_ids[i])
                ppe_track_ids.append(track_ids[i])
        person_ids = [
            item for item in person_ids if item not in person_track_ids] + person_track_ids

        f.write(
            f'{total_time}: {assign_ppe_to_person(ppe_boxes,[labels[x] for x in ppe_class_ids], person_bbox,[person_ids.index(x) for x in person_track_ids])}\n\n')
        
        draw_bounding_box(image, ppe_boxes, ppe_confidences, labels, ppe_class_ids, colors)

        elapsed_time = (time.time() - start_time)
        total_time += 1.0/frame_rate
        if video is not None:
            fps = 1.0 / elapsed_time
            fps_text = f"FPS: {fps:.2f}"
            # cv2.putText(image, fps_text, (10, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite("./"+ output_path+"/"+tail,image)

        if video is not None:
            success, image = video.read()
        else:
            success = False
    f.close()
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str,
                        help='input path, image or mp4')
    parser.add_argument('--output_path', type=str, help='output path')
    # parser.add_argument('--device', type=str, help='cuda device')
    args = parser.parse_args()

    detect_and_draw(args.input_path, args.output_path, 'cuda:'+'0')
