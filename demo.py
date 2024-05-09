import os
from typing import Optional
import numpy as np
import yolov5
import cv2
from PIL import Image
from collections import defaultdict
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import pandas as pd
import sources.Controllers.config as cfg
from sources.Controllers import utils

SAVE_DIR = "results"
CONTENT_MODEL = yolov5.load(cfg.CONTENT_MODEL_PATH)
# Set conf and iou threshold -> Remove overlap and low confident bounding boxes
CONTENT_MODEL.conf = cfg.CONF_CONTENT_THRESHOLD
CONTENT_MODEL.iou = cfg.IOU_CONTENT_THRESHOLD

# CORNER_MODEL.conf = cfg.CONF_CORNER_THRESHOLD
# CORNER_MODEL.iou = cfg.IOU_CORNER_THRESHOLD

# Config directory
UPLOAD_FOLDER = cfg.UPLOAD_FOLDER
SAVE_DIR = cfg.SAVE_DIR
CROP_DIR = cfg.CROP_DIR
FACE_CROP_DIR = cfg.FACE_DIR
SHARPEN_DIR = cfg.SHARPEN_DIR

""" Recognizion detected parts in ID """
config = Cfg.load_config_from_name(
    "vgg_transformer"
)  # OR vgg_transformer -> acc || vgg_seq2seq -> time
# config = Cfg.load_config_from_file(cfg.OCR_CFG)
config['weights'] = cfg.OCR_MODEL_PATH
config["cnn"]["pretrained"] = False
config["device"] = cfg.DEVICE
config["predictor"]["beamsearch"] = False
detector = Predictor(config)

def demo_images(path_id=None):
    # Check if the specified image directory exists
    if not os.path.isdir(path_id):
        print(f"Error: Image directory '{path_id}' does not exist.")
        return

    # Create the output directory if it doesn't exist
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Get a list of image files in the specified directory
    image_files = [f for f in os.listdir(path_id) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
    all_detected_list = []
    # Iterate over each image file in the directory
    count = 0
    for img_file in image_files:
        count += 1
        print(f"Processing image {count}/{len(image_files)}: {img_file}")
        img = os.path.join(path_id, img_file)

        # Extract number sequence from image filename
        image_id = ''.join(c for c in img_file if c.isdigit())

        CORNER_MODEL = yolov5.load(cfg.CORNER_MODEL_PATH)
        CORNER = CORNER_MODEL(img)
        predictions = CORNER.pred[0]
        boxes_dict = {}

        for i, value in enumerate(predictions[:, 5]):
            if value == 0.:
                boxes_dict["top_left"] = predictions[i, :4]
            elif value == 1.:
                boxes_dict["top_right"] = predictions[i, :4]
            elif value == 2.:
                boxes_dict["bottom_left"] = predictions[i, :4]
            elif value == 3.:
                boxes_dict["bottom_right"] = predictions[i, :4]

        # Check the number of detected corners
        num_corners = len(boxes_dict)
        if num_corners < 3:
            # Handle the case where less than 4 corners are detected
            print(f"Skipping image '{img_file}': Detected only {num_corners} corners!")
            continue

        # Draw bounding boxes around corners:
        drawImg = cv2.imread(img)
        for corner_name, box_info in boxes_dict.items():
            left, top, right, bottom = map(int, box_info)
            cv2.rectangle(drawImg, (left, top), (right, bottom), (0, 0, 255), 2) # Red rectangle
            # Draw center points
            center_x, center_y = utils.get_center_point(box_info)
            center_x, center_y = int(center_x), int(center_y)
            cv2.circle(drawImg, (center_x, center_y), 5, (0, 255, 0), -1) # Green circle

        # Save the output image with 4 corners bbox
        # output_path = os.path.join(SAVE_DIR, f"{os.path.splitext(img_file)[0]}_output.jpg")
        # cv2.imwrite(output_path, drawImg)
        # print(f"Saved output image to {output_path}")

        IMG = cv2.imread(img)
        center_points_dict = {corner_name: utils.get_center_point(box_info) for corner_name, box_info in boxes_dict.items()}

        """ Temporary fixing """
        aligned = utils.align_image(IMG, center_points_dict)
        # Convert from OpenCV to PIL format
        aligned = Image.fromarray(aligned)
        # Save the aligned image
        # aligned_output_path = os.path.join(SAVE_DIR, f"{os.path.splitext(img_file)[0]}_aligned.jpg")
        # aligned.save(aligned_output_path)
        # print(f"Saved aligned image to {aligned_output_path}")

        # content model to detect boxes
        CONTENT = CONTENT_MODEL(aligned)
        # CONTENT.save(save_dir='results/')
        predictions = CONTENT.pred[0]
        # print("Predictions:")
        # print(predictions)
        categories = predictions[:, 5].tolist()  # Class
        if len(categories) == 0:
            print(f"No categories detected in image '{img_file}'")
            continue
        boxes = predictions[:, :4].tolist()
        if len(boxes) == 0:
            print(f"No boxes detected in image '{img_file}'")
            continue
        """ Non Maximum Suppression """
        boxes, categories = utils.non_max_suppression_fast(np.array(boxes), categories, 0.2)
        # print("Box information:")
        # print(boxes)
        # print("Category information:")
        # print(categories)
        if not os.path.isdir(SAVE_DIR):
            os.mkdir(SAVE_DIR)
        else:
            for f in os.listdir(SAVE_DIR):
                os.remove(os.path.join(SAVE_DIR, f))

        # Save the output images with boxes
        for index, box in enumerate(boxes):
            left, top, right, bottom = box
            cropped_image = aligned.crop((left, top, right, bottom))
            cropped_image.save(os.path.join(SAVE_DIR, f"{index}.jpg"))

        # Collecting all detected parts
        FIELDS_DETECTED = defaultdict(list)
        image_files = sorted(os.listdir(SAVE_DIR), reverse=True)  # Sort in reverse order
        reversed_categories = categories[::-1]
        for idx, img_crop in enumerate(image_files):
            img_path = os.path.join(SAVE_DIR, img_crop)
            img_ = Image.open(img_path)
            img = cv2.imread(img_path)

            s = detector.predict(img_)
            label = reversed_categories[idx]  # Assuming labels is a list corresponding to boxes

            #save cropped images of content infomation
            # if (label in (5.0, 6.0)):
            #     img_.save(os.path.join(CROP_DIR, f"{img_file}_{idx}.jpg"))

            FIELDS_DETECTED[label].append(s)
        # Combine fields with the same label

        for key, values in FIELDS_DETECTED.items():
            if len(values) > 1:
                combined_value = ', '.join(values)
                FIELDS_DETECTED[key] = combined_value
            else:
                FIELDS_DETECTED[key] = values[0]

        print("FIELDS_DETECTED values:")
        for key, value in FIELDS_DETECTED.items():
            print(f"{key}: {value}")
        # Convert defaultdict to regular dict
        FIELDS_DETECTED = dict(FIELDS_DETECTED)
        FIELDS_DETECTED['image_id'] = image_id
        all_detected_list.append(FIELDS_DETECTED)
        response = {"data": list(FIELDS_DETECTED.values())}

    df_combined = pd.DataFrame(all_detected_list)
    csv_file_path = "info_detected_cccd_v3_20k.csv"
    df_combined.to_csv(csv_file_path, index=False)

demo_images(path_id='cccd_20k_images')
