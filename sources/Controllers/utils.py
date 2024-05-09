import cv2
import numpy as np


def class_Order(boxes, categories):
    Z = []
    # Z = [x for _,x in sorted(zip(categories, boxes))]
    cate = np.argsort(categories)
    for index in cate:
        Z.append(boxes[index])

    return Z


def non_max_suppression_fast(boxes, labels, overlapThresh):
    print("Original labels: " + str(labels))
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

        # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 1]
    y1 = boxes[:, 0]
    x2 = boxes[:, 3]
    y2 = boxes[:, 2]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(x2)
    print("Index:" + str(idxs)  + " " + str(len(idxs)))

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(
            idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
        )

    # return only the bounding boxes that were picked using the
    # integer data type
    # print("Pick info: " + str(pick) + " " + str(len(pick)))
    final_labels = [labels[k] for k in pick]
    #print final labels info
    # print("Final labels: " + str(final_labels))
    final_boxes = boxes[pick].astype("int")
    # print("Final boxes: " + str(final_boxes))
    return final_boxes, final_labels


def get_center_point(box):
    left, top, right, bottom = box
    return left + ((right - left) // 2), top + (
        (bottom - top) // 2
    )  # (x_c, y_c) # Need to fix bottom_left and bottom_right


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def find_miss_corner(coordinate_dict):
    position_name = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    position_index = np.array([0, 0, 0, 0])

    for name in coordinate_dict.keys():
        if name in position_name:
            position_index[position_name.index(name)] = 1

    index = np.argmin(position_index)

    return index


def calculate_missed_coord_corner(coordinate_dict):
    thresh = 0

    index = find_miss_corner(coordinate_dict)
    # calculate missed corner coordinate
    # case 1: missed corner is "top_left"
    if index == 0:
        midpoint = np.add(coordinate_dict['top_right'], coordinate_dict['bottom_left']) / 2
        y = 2 * midpoint[1] - coordinate_dict['bottom_right'][1] - thresh
        x = 2 * midpoint[0] - coordinate_dict['bottom_right'][0] - thresh
        coordinate_dict['top_left'] = (x, y)
    elif index == 1:  # "top_right"
        midpoint = np.add(coordinate_dict['top_left'], coordinate_dict['bottom_right']) / 2
        y = 2 * midpoint[1] - coordinate_dict['bottom_left'][1] - thresh
        x = 2 * midpoint[0] - coordinate_dict['bottom_left'][0] - thresh
        coordinate_dict['top_right'] = (x, y)
    elif index == 2:  # "bottom_left"
        midpoint = np.add(coordinate_dict['top_left'], coordinate_dict['bottom_right']) / 2
        y = 2 * midpoint[1] - coordinate_dict['top_right'][1] - thresh
        x = 2 * midpoint[0] - coordinate_dict['top_right'][0] - thresh
        coordinate_dict['bottom_left'] = (x, y)
    elif index == 3:  # "bottom_right"
        midpoint = np.add(coordinate_dict['bottom_left'], coordinate_dict['top_right']) / 2
        y = 2 * midpoint[1] - coordinate_dict['top_left'][1] - thresh
        x = 2 * midpoint[0] - coordinate_dict['top_left'][0] - thresh
        coordinate_dict['bottom_right'] = (x, y)

    return coordinate_dict

def align_image(image, pts):
    if len(pts) <3:
        raise Exception("Not enough points to align")
    image = np.asarray(image)
    if len(pts) == 3:
        pts = calculate_missed_coord_corner(pts)
        print(pts)
    # rect = order_points(pts)
    tl = pts.get('top_left')
    tr = pts.get('top_right')
    br = pts.get('bottom_right')
    bl = pts.get('bottom_left')
    # print("4 locations:")
    # print(tl, tr, bl, br)
    # image_with_points = image.copy()
    cv2.circle(image, tuple(map(int, tl)), 5, (0, 0, 255), -1)  # Red circle for top-left
    cv2.circle(image, tuple(map(int, tr)), 5, (0, 255, 0), -1)  # Green circle for top-right
    cv2.circle(image, tuple(map(int, br)), 5, (255, 0, 0), -1)  # Blue circle for bottom-right
    cv2.circle(image, tuple(map(int, bl)), 5, (255, 255, 0), -1)  # Cyan circle for bottom-left
    cv2.imwrite("image_before_aligned.jpg", image)
    source_points = np.float32([tl, tr, br, bl])
    crop = perspective_transform(image, source_points)
    return crop
# focus and transform the cccd image into a rectangle image
def perspective_transform(image, source_points):
    dest_points = np.float32([[0, 0], [525, 0], [525, 310], [0, 310]])
    M = cv2.getPerspectiveTransform(source_points, dest_points)
    dst = cv2.warpPerspective(image, M, (525, 310))

    return dst


def four_point_transform(image, pts):
    image = np.asarray(image)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


# def getMissingCorner(categories, boxes): # boxes: top_left, top_right, bottom_left, bottom_right
# 	if 0 not in categories: # Missing top_left
# 		delta_vertical = boxes[3][2] - boxes[1][2]
# 		delta_horizon = boxes[3][3] - boxes[2][3]
# 		x_miss =
