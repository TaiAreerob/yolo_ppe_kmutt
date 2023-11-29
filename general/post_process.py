from general.general import calculate_iou


def assign_ppe_to_person(ppe_bboxes, class_ids, person_bboxes, track_ids):
    assignments = {}

    for i, ppe_bbox in enumerate(ppe_bboxes):
        max_overlap = 0
        max_overlap_person = None

        for idx, person_bbox in enumerate(person_bboxes):
            overlap = calculate_iou(ppe_bbox, person_bbox)

            if overlap > max_overlap:
                max_overlap = overlap
                max_overlap_person = track_ids[idx]

        if max_overlap_person in assignments:
            assignments[max_overlap_person].append(class_ids[i])
        else:
            assignments[max_overlap_person] = [class_ids[i]]

    assignment_list = [{'person': person, 'ppe': ppe}
                       for person, ppe in assignments.items()]

    return assignment_list
