from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import cv2
import matplotlib.pyplot as plt
import numpy as np

model = YOLO("yolo11s-seg.pt")

# model_s_best = YOLO("runs/segment/train3/weights/best.pt") #>>uncomment if continuing training from existing model
results = model_s_best.train(data="data.yaml", epochs=5, imgsz=640)
# # print(results)

metrics = model.val(data="data.yaml", save_json=True)  # assumes `model` has been loaded
print(metrics.box.map)  # mAP50-95
print(metrics.box.map50)  # mAP50
print(metrics.box.map75)  # mAP75
print(metrics.box.maps)  # list of mAP50-95 for each category


prediction = model_s_best.predict(source="68train.jpg",save=True, save_txt=True)


image = cv2.imread('68train.jpg')

img_result = image.copy()
for result in prediction:
    masks_array = result.masks.xy
    for mask_array in masks_array:
        points = np.array(mask_array, dtype=np.int32)
        pts = points.reshape((-1, 1, 2))
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        blurred = cv2.GaussianBlur(image, (25, 25), 0)
        img_result[mask == 255] = blurred[mask == 255]


cv2.imwrite('output.jpg', img_result)
# cv2.imwrite('_model_s_best.jpg', img_result)
