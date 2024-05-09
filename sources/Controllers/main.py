import os
from typing import Optional

import torch
import databases
import numpy as np
import yolov5
import cv2
from fastapi import Depends, File, Form, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from pylibsrtp import Session
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

import sources.Controllers.config as cfg
from sources import app, templates
from sources.Controllers import utils
from sources.Models import models
from sources.Models.database import SQLALCHEMY_DATABASE_URL, SessionLocal, engine
from sources.Models.models import Feedback
from collections import defaultdict


""" ---- Setup ---- """
# Init Database
database = databases.Database(SQLALCHEMY_DATABASE_URL)
models.Base.metadata.create_all(bind=engine)


async def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Startup database server before start app
@app.on_event("startup")
async def startup_database():
    await database.connect()


# Shutdown database sever after closed app
@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()


# Init yolov5 model
CORNER_MODEL = yolov5.load(cfg.CORNER_MODEL_PATH)
CONTENT_MODEL = yolov5.load(cfg.CONTENT_MODEL_PATH)
FACE_MODEL = yolov5.load(cfg.FACE_MODEL_PATH)

# Set conf and iou threshold -> Remove overlap and low confident bounding boxes
CONTENT_MODEL.conf = cfg.CONF_CONTENT_THRESHOLD
CONTENT_MODEL.iou = cfg.IOU_CONTENT_THRESHOLD

# CORNER_MODEL.conf = cfg.CONF_CORNER_THRESHOLD
# CORNER_MODEL.iou = cfg.IOU_CORNER_THRESHOLD

# Config directory
UPLOAD_FOLDER = cfg.UPLOAD_FOLDER
SAVE_DIR = cfg.SAVE_DIR
FACE_CROP_DIR = cfg.FACE_DIR
SHARPEN_DIR = cfg.SHARPEN_DIR

""" ---- ##### -----"""


class feedback_Request(BaseModel):
    content: str
    rating: int

    class Config:
        orm_mode = True


class contact_Request(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    message: str

    class Config:
        orm_mode = True


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


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/home")
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/id_card")
async def id_extract_page(request: Request):
    return templates.TemplateResponse("idcard.html", {"request": request})


@app.get("/ekyc")
async def ekyc_page(request: Request):
    return templates.TemplateResponse("ekyc.html", {"request": request})


@app.get("/feedback")
async def feedback_page(request: Request):
    return templates.TemplateResponse("feedback.html", {"request": request})


@app.get("/contact")
async def contact_page(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request})


@app.post("/uploader")
async def upload(file: UploadFile = File(...)):
    INPUT_IMG = os.listdir(UPLOAD_FOLDER)
    if INPUT_IMG is not None:
        for uploaded_img in INPUT_IMG:
            os.remove(os.path.join(UPLOAD_FOLDER, uploaded_img))

    file_location = f"./{UPLOAD_FOLDER}/{file.filename}"
    contents = await file.read()
    with open(file_location, "wb") as f:
        f.write(contents)

    # Validating file
    INPUT_FILE = os.listdir(UPLOAD_FOLDER)[0]
    if INPUT_FILE == "NULL":
        os.remove(os.path.join(UPLOAD_FOLDER, INPUT_FILE))
        error = "No file selected!"
        return JSONResponse(status_code=403, content={"message": error})
    elif INPUT_FILE == "WRONG_EXTS":
        os.remove(os.path.join(UPLOAD_FOLDER, INPUT_FILE))
        error = "This file is not supported!"
        return JSONResponse(status_code=404, content={"message": error})

    # return {"Filename": file.filename}
    return await extract_info()


@app.post("/extract")
# @app.api_route("/extract", methods=["GET", "POST"])
async def extract_info(ekyc=False, path_id=None):
    """Check if uploaded image exist"""
    if not os.path.isdir(cfg.UPLOAD_FOLDER):
        os.mkdir(cfg.UPLOAD_FOLDER)

    INPUT_IMG = os.listdir(UPLOAD_FOLDER)
    if INPUT_IMG is not None:
        if not ekyc:
            img = os.path.join(UPLOAD_FOLDER, INPUT_IMG[0])
        else:
            img = path_id
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
        
    print("Boxes dict:")
    print(boxes_dict)


    #Check the number of detected corners
    num_corners = len(boxes_dict)
    if num_corners < 3:
        # Handle the case where less than 4 corners are detected
        if num_corners == 0:
            error = "Detected = 0!"
        else: 
            error = f"Deteced only {num_corners} corners!"
        return JSONResponse(status_code=401, content={"message": error})
    
    # Print corner information and structure
    print("Corner information:")
    for corner_name, box_info in boxes_dict.items():
        print(f"{corner_name}: {box_info}")

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
    output_path = 'output.jpg'
    cv2.imwrite(output_path, drawImg)
    print(f"Saved output image to {output_path}")

    # IMG = Image.open(img)
    # IMG.save("output_image.jpg")
    IMG = cv2.imread(img)
    # cv2.imwrite("output_image1.jpg", IMG)
    center_points_dict = {corner_name: utils.get_center_point(box_info) for corner_name, box_info in boxes_dict.items()}

    print("Center points information:")
    for corner_name, center_point in center_points_dict.items():
        print(f"Center Point for {corner_name}: {center_point}")
    """ Temporary fixing """
    aligned = utils.align_image(IMG, center_points_dict)
    # Convert from OpenCV to PIL format
    aligned = Image.fromarray(aligned)
    aligned.save('aligned.jpg')
    # CORNER.show()

    CONTENT = CONTENT_MODEL(aligned)
    # CONTENT.save(save_dir='results/')
    predictions = CONTENT.pred[0]
    print("Prediction information:")
    print(predictions)
    categories = predictions[:, 5].tolist()  # Class
    # if 7 not in categories:
    #     if len(categories) < 9:
    #         error = "Missing fields! Detecting content failed!"
    #         return JSONResponse(status_code=402, content={"message": error})
    # elif 7 in categories:
    #     if len(categories) < 10:
    #         error = "Missing fields! Detecting content failed!"
    #         return JSONResponse(status_code=402, content={"message": error})

    boxes = predictions[:, :4].tolist()
    print("box before NMS:")
    print(boxes)
    """ Non Maximum Suppression """
    boxes, categories = utils.non_max_suppression_fast(np.array(boxes), categories, 0.2)
    # boxes = utils.class_Order(boxes, categories)  # x1, x2, y1, y2
    print("Box information:")
    print(boxes)
    print("Category information:")
    print(categories)
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

        # Create a filter matrix
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

        # Apply the filter matrix to the image
        sharpened = cv2.filter2D(img, -1, kernel)

        # Convert NumPy array to PIL image
        sharpened_pil = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
        sharpened_pil.save(SHARPEN_DIR  + f"/{idx}.jpg")
        s = detector.predict(img_)
        label = reversed_categories[idx]  # Assuming labels is a list corresponding to boxes
        FIELDS_DETECTED[label].append(s)

    # Print the values of FIELDS_DETECTED


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

    response = {"data": list(FIELDS_DETECTED.values())}

    response = jsonable_encoder(response)
    return JSONResponse(content=response)


@app.post("/download")
async def download(file: str = Form(...)):
    if file != "undefined":
        noti = "Download file successfully!"
        return JSONResponse(status_code=201, content={"message": noti})
    else:
        error = "No file to download!"
        return JSONResponse(status_code=405, content={"message": error})


@app.post("/feedback")
async def save_feedback(
    content: str = Form(...), rating: int = Form(...), db: Session = Depends(get_db)
):
    feedback = Feedback()
    feedback.content = content
    feedback.rating = rating
    db.add(feedback)
    db.commit()

    response = {"code": "200", "content": "save successfully"}

    return JSONResponse(content=response)


@app.post("/contact")
async def contact(request: contact_Request):
    # print(request.name)
    pass


@app.post("/ekyc/uploader")
async def get_id_card(id: UploadFile = File(...), img: UploadFile = File(...)):
    INPUT_IMG = os.listdir(UPLOAD_FOLDER)
    if INPUT_IMG is not None:
        for uploaded_img in INPUT_IMG:
            os.remove(os.path.join(UPLOAD_FOLDER, uploaded_img))

    id_location = f"./{UPLOAD_FOLDER}/{id.filename}"
    id_contents = await id.read()

    with open(id_location, "wb") as f:
        f.write(id_contents)

    img_location = f"./{UPLOAD_FOLDER}/{img.filename}"
    img_contents = await img.read()
    with open(img_location, "wb") as f_:
        f_.write(img_contents)

    # Validating file
    INPUT_FILE = os.listdir(UPLOAD_FOLDER)
    if "NULL_1" in INPUT_FILE and "NULL_2" not in INPUT_FILE:
        for uploaded_img in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER, uploaded_img))
        error = "Missing ID card image!"
        return JSONResponse(status_code=410, content={"message": error})
    elif "NULL_2" in INPUT_FILE and "NULL_1" not in INPUT_FILE:
        for uploaded_img in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER, uploaded_img))
        error = "Missing person image!"
        return JSONResponse(status_code=411, content={"message": error})
    elif "NULL_1" in INPUT_FILE and "NULL_2" in INPUT_FILE:
        for uploaded_img in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER, uploaded_img))
        error = "Missing ID card and person images!"
        return JSONResponse(status_code=412, content={"message": error})
    else:
        id_name = id.filename.split(".")
        new_id_name = f"./{UPLOAD_FOLDER}/id.{id_name[-1]}"
        os.rename(id_location, new_id_name)
        img_name = img.filename.split(".")
        new_img_name = f"./{UPLOAD_FOLDER}/person.{img_name[-1]}"
        os.rename(img_location, new_img_name)

    FACE = FACE_MODEL(new_img_name)
    predictions = FACE.pred[0]
    categories = predictions[:, 5].tolist()  # Class
    if 0 not in categories:
        error = "No face detected!"
        return JSONResponse(status_code=413, content={"message": error})
    elif categories.count(0) > 1:
        error = "Multiple faces detected!"
        return JSONResponse(status_code=414, content={"message": error})

    boxes = predictions[:, :4].tolist()

    """ Non Maximum Suppression """
    boxes, categories = utils.non_max_suppression_fast(np.array(boxes), categories, 0.7)

    if not os.path.isdir(FACE_CROP_DIR):
        os.mkdir(FACE_CROP_DIR)
    else:
        for f in os.listdir(FACE_CROP_DIR):
            os.remove(os.path.join(FACE_CROP_DIR, f))

    FACE_IMG = Image.open(new_img_name)
    # left, top, right, bottom = boxes[0]
    cropped_image = FACE_IMG.crop((boxes[0]))
    cropped_image.save(os.path.join(FACE_CROP_DIR, "face_crop.jpg"))

    return await extract_info(ekyc=True, path_id=new_id_name)
