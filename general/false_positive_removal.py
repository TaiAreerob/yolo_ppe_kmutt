from general.general import are_near, calculate_iou, is_inside, middle_point


def get_bounding_box_inside_person(boxes, person_bbox, confidences, class_ids, threshold=[0.5, 0.5, 0.5, 0.5]):
    boxes_t, confidences_t, class_ids_t = [[], [], []]
    for idx in range(len(boxes)):
        for p in person_bbox:
            if is_inside(p, boxes[idx], threshold[class_ids[idx]]):
                boxes_t.append(boxes[idx])
                confidences_t.append(confidences[idx])
                class_ids_t.append(class_ids[idx])
                break
    return boxes_t, confidences_t, class_ids_t


def get_bounding_box_nearby_bodypart(kpts, boxes, confidences, class_ids, threshold=[0.5, 0.5, 0.5, 0.5]):
    boxes_t, confidences_t, class_ids_t = [[], [], []]
    for i in range(len(boxes)):
        for kpt in kpts:
            if (class_ids[i] == 0 and are_near(boxes[i], middle_point([kpt[10]]), threshold[0])) or (class_ids[i] == 0 and are_near(boxes[i], middle_point([kpt[9]]), threshold[0])) or (class_ids[i] == 1 and are_near(boxes[i], middle_point([kpt[0], kpt[1], kpt[2], kpt[3]]), threshold[1])) or (class_ids[i] == 2 and are_near(boxes[i], middle_point([kpt[15]]), threshold[2])) or (class_ids[i] == 2 and are_near(boxes[i], middle_point([kpt[16]]), threshold[2])) or (class_ids[i] == 3 and are_near(boxes[i], middle_point([kpt[5], kpt[6], kpt[11], kpt[12]]), threshold[3])):
                boxes_t.append(boxes[i])
                confidences_t.append(confidences[i])
                class_ids_t.append(class_ids[i])
                break
    return boxes_t, confidences_t, class_ids_t


def check_highest_list_iou(bboxes):
    best_iou = 0
    best_index = -1

    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            iou = calculate_iou(bboxes[i]['ppe_box'], bboxes[j]['ppe_box'])
            if iou > best_iou:
                best_iou = iou
                best_index = bboxes[j]['ppe_id']

    return best_iou, best_index


def check_highest_iou(bboxes, bbox):
    best_iou = 0
    best_index = 0

    for i, bb in enumerate(bboxes):
        iou = calculate_iou(bbox, bb['ppe_box'])
        if iou > best_iou:
            best_iou = iou
            best_index = bb['ppe_id']

    return best_iou, best_index


def check_lowest_conf(ppe_object, ppe_conf):
    lowest_conf = 1
    index = 0
    for ppe in ppe_object:
        if ppe_conf[ppe['ppe_id']] < lowest_conf:
            lowest_conf = ppe_conf[ppe['ppe_id']]
            index = ppe['ppe_id']
    return lowest_conf, index


def filter_ppe(person_boxes, ppe_boxes, ppe_conf, ppe_class, max_class_amount):
    # assign ppe to person -> [][] person x ppe
    person = [[[] for x in range((max(ppe_class) if len(
        ppe_class) > 0 else 0)+1)] for x in person_boxes]
    for i, ppe in enumerate(ppe_boxes):
        highest_iou = 0
        highest_index = -1
        for j, _person in enumerate(person_boxes):
            if calculate_iou(_person, ppe) > highest_iou:
                highest_iou = calculate_iou(_person, ppe)
                highest_index = j
        if highest_index != -1:
            person[highest_index][ppe_class[i]].append(
                {'ppe_id': i, 'ppe_box': ppe})

    # filter out the over detected ppe
    for i, _person in enumerate(person):
        for j, _ppe_class in enumerate(_person):
            if len(_ppe_class) <= max_class_amount[j]:
                continue
            else:
                temp_ppe = []
                for k, ppe in enumerate(_ppe_class):
                    if len(temp_ppe) < max_class_amount[j]:
                        temp_ppe.append(ppe)
                    else:
                        if check_highest_list_iou(temp_ppe)[0] > 0.2 and check_highest_iou(temp_ppe, ppe['ppe_box'])[0] < check_highest_list_iou(temp_ppe)[0] and check_highest_iou(temp_ppe, ppe['ppe_box'])[1] != check_highest_list_iou(temp_ppe)[1]:
                            index = check_highest_list_iou(temp_ppe)[1]
                            temp_ppe = list(
                                filter(lambda x: x['ppe_id'] != index, temp_ppe))
                            temp_ppe.append(ppe)
                        elif (check_highest_iou(temp_ppe, ppe['ppe_box'])[0] < 0.2 and check_lowest_conf(temp_ppe, ppe_conf)[0] < ppe_conf[ppe['ppe_id']]):
                            index = check_lowest_conf(temp_ppe, ppe_conf)[1]
                            temp_ppe = list(
                                filter(lambda x: x['ppe_id'] != index, temp_ppe))
                            temp_ppe.append(ppe)
                person[i][j] = temp_ppe

    _ppe_boxes, _ppe_conf, _ppe_class = [[], [], []]
    for i, _person in enumerate(person):
        for j, __ppe_class in enumerate(_person):
            for k, ppe in enumerate(__ppe_class):
                _ppe_boxes.append(ppe_boxes[ppe['ppe_id']])
                _ppe_conf.append(ppe_conf[ppe['ppe_id']])
                _ppe_class.append(ppe_class[ppe['ppe_id']])
    return _ppe_boxes, _ppe_conf, _ppe_class
