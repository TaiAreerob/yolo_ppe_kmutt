import cv2


def draw_bounding_box(image, boxes, confidences, labels, class_ids, colors):
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        label = f"{labels[class_ids[i]]}"
        tl = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c1, c2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(
            image, c1, c2, colors[class_ids[i]], thickness=tl, lineType=cv2.LINE_AA)
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, colors[class_ids[i]], -1, cv2.LINE_AA)
        cv2.putText(image, label, (c1[0], c1[1] - 2), 0,  tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
