from icevision.models.inference import process_bbox_predictions
from icevision.all import *
from PIL import Image

img_path = "/home.stud/kuntluka/dataset/carries_dataset/images/1418.png"
img = Image.open(img_path)
# print(preds[7].ground_truth.common.filepath)

tr = [*tfms.A.resize_and_pad(1024), tfms.A.Normalize()]
# BaseRecord
proc = process_bbox_predictions(Prediction(BaseRecord()), img, tr)