PORT = 8080
CONF_CONTENT_THRESHOLD = 0.45
IOU_CONTENT_THRESHOLD = 0.7

CONF_CORNER_THRESHOLD = 0.2
IOU_CORNER_THRESHOLD = 0.5

CORNER_MODEL_PATH = "sources/Database/OCR/weights/new_cccd_corner_v1.pt"
CONTENT_MODEL_PATH = "sources/Database/OCR/weights/cccd-new-content-v1.pt"
FACE_MODEL_PATH = "sources/Database/OCR/weights/face.pt"
OCR_MODEL_PATH = "sources/Database/OCR/weights/transformerocr_cccd_v3.3.pth"
# OCR_CFG = 'sources/Database/OCR/config/seq2seq_config.yml'
DEVICE = "cpu"  # or "cuda:0" if using GPU
# Config directory
UPLOAD_FOLDER = "sources/Database/uploads"
SAVE_DIR = "align_images_v3"
CROP_DIR = "cropped_images_v1"
CROP_NAME_DIR = "cropped_name"
SHARPEN_DIR = "sources/static/sharpen"
FACE_DIR = "sources/static/face"
CORNER_SAVE_DIR = "sources/static/resutls/corners"
