import cv2
import numpy as np
from PIL import Image

def blur_documents(image, predictions, threshold=0.5):
    img = np.array(image).copy()
    masks = predictions['masks'] > threshold

    for mask in masks:
        mask = mask.squeeze().cpu().numpy().astype(np.uint8)
        blurred = cv2.GaussianBlur(img, (21, 21), 0)
        img[mask == 1] = blurred[mask == 1]

    return Image.fromarray(img)
