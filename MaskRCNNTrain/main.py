import json
import torch
import torchvision
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.models.detection import maskrcnn_resnet50_fpn 
import numpy as np



import modelinstance
import backbone
import DocumentSegmentationDataset
import TrainUtils
import BluringUtils
import cv2

NUM_CLASSES = 13 # background, passport, id_card, paper_document
# DOCUMENT_CLASSES = {0, 4, 10}
DOCUMENT_CLASSES = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0}
EPOCHS = 20
IMAGE_DIR = "train"
ANNOTATIONS_FILE = "train/_annotations.coco.json"
MODEL_PATH = "exported_model.pth"
PREDICTION_IMAGE = "download.jpg"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

if __name__ == "__main__":
    # 1. Load annotations
    with open(ANNOTATIONS_FILE, 'r') as f:
        label_data = json.load(f)


    transform_pipeline = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    # 2. Load dataset
    dataset = DocumentSegmentationDataset.DocumentSegmentationDataset(
        image_dir=IMAGE_DIR,
        label_data=label_data,
        transforms=transform_pipeline
    )

    # # 3. Initialize model
    model = modelinstance.build_instance_seg_model(NUM_CLASSES).to(DEVICE)  #>>comment if continuing training from a saved model

    # model = maskrcnn_resnet50_fpn(num_classes=NUM_CLASSES)  # set NUM_CLASSES as used in training #>> uncomment if continuing training from a saved model
    # model.load_state_dict(torch.load("exported_model.pth", map_location=torch.device("cpu"))) #>> uncomment if continuing training from a saved model
    model.eval()

    # 4. Train model
    print("Training model...")
    TrainUtils.train(model, dataset, num_epochs=EPOCHS)

    # 5. Export model
    print(f"Saving trained model to {MODEL_PATH}")
    torch.save(model.state_dict(), MODEL_PATH)


    # 6. Load image and run prediction
    print("Running prediction on sample image...")
    model.eval()
    image = Image.open(PREDICTION_IMAGE)    #.convert("RGB")
    image_tensor = torchvision.transforms.ToTensor()(image).unsqueeze(0).to(DEVICE)
    # print(image_tensor)

    with torch.no_grad():
        outputs = model(image_tensor)
    preds = outputs[0]

    # for output in outputs:
    #     print(output)

    # 7. Blur documents

    image = cv2.imread('download.jpg')

    threshold = 0.8
    scores = preds['scores']
    labels = preds['labels']
    masks = preds['masks']
    print(scores)
    print(scores > threshold)

    # Step 3: Filter predictions
    keep = (scores > threshold) & (labels.cpu().numpy()[:, None] == list(DOCUMENT_CLASSES)).any(axis=1)
    filtered_masks = masks[keep].squeeze(1).cpu().numpy()
    print(filtered_masks)
    # Step 4: Load image and blur
    image = cv2.imread(PREDICTION_IMAGE)
    img_result = image.copy()

    for mask in filtered_masks:
        print('mask')
        mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            mask_poly = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask_poly, [cnt], 255)
            blurred = cv2.GaussianBlur(image, (25, 25), 0)
            img_result[mask_poly == 255] = blurred[mask_poly == 255]

    cv2.imwrite('output.jpg', img_result)
    print("Blurred output saved as output.jpg")


    cv2.imwrite('output.jpg', img_result)

    print("Blurred output saved as output.jpg")
