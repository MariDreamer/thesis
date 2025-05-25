import sys
import os
import boto3
import zipfile
import shutil

S3_BUCKET = 'sensitivefilebluringbucket'
S3_KEY = 'python.zip'
EFS_ZIP_PATH = '/mnt/efs/python.zip'
EXTRACT_DIR = '/mnt/efs/python'

# if os.path.exists(EXTRACT_DIR):
#     sys.path.insert(0, EXTRACT_DIR)

# os.environ["LD_LIBRARY_PATH"] = "/mnt/efs/python/torch/lib:" + os.environ.get("LD_LIBRARY_PATH", "")


NUM_CLASSES = 13
DOCUMENT_CLASSES = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
MODEL_PATH = "model/rcnn_model_document_recognition.pt"  # Assumes model is in your Lambda Layer under /opt/


def download_zip_if_missing():
    if not os.path.exists(EFS_ZIP_PATH):
      print("Downloading python.zip from S3...")
      s3 = boto3.client('s3')
      with open(EFS_ZIP_PATH, 'wb') as f:
          s3.download_fileobj(S3_BUCKET, S3_KEY, f)
      print("Download complete.")
    else:
      print("python.zip already exists.")

    # sys.path.insert(0, EFS_ZIP_PATH)
    if not os.path.exists(EXTRACT_DIR):
        with zipfile.ZipFile(EFS_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
    sys.path.insert(0, EXTRACT_DIR)

def delete_folder_from_efs(folder_path):
    if os.path.exists(folder_path):
        if os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            print(f"Deleted folder: {folder_path}")
        else:
            print(f"{folder_path} exists but is not a directory.")
    else:
        print(f"{folder_path} does not exist.")

def lambda_handler(event, context):
    # delete_folder_from_efs("/mnt/efs/python")
    download_zip_if_missing()

    # lazy imports, after python folder exists in EFS
    import json
    import torch
    import torchvision
    from PIL import Image
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    import numpy as np
    import io
    import base64
    import cv2

        
        
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model once when the Lambda container is initialized
    model = maskrcnn_resnet50_fpn(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)

    try:
        # Read and decode binary body
        is_base64 = event.get("isBase64Encoded", False)
        body = event["body"]
        if is_base64:
            image_data = base64.b64decode(body)
        else:
            image_data = body.encode("latin1")  # Fallback if not base64 (rare)

        # Load image into PIL and CV2
        image_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_tensor = torchvision.transforms.ToTensor()(image_pil).unsqueeze(0).to(DEVICE)

        # Run model inference
        with torch.no_grad():
            outputs = model(image_tensor)
        preds = outputs[0]

        # Prepare for blurring
        image_array = np.array(image_pil)
        image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        result_img = image_cv.copy()

        threshold = 0.5
        scores = preds["scores"].cpu().numpy()
        labels = preds["labels"].cpu().numpy()
        masks = preds["masks"].cpu().numpy()

        keep = (scores > threshold) & np.isin(labels, list(DOCUMENT_CLASSES))
        filtered_masks = masks[keep].squeeze(1)

        for mask in filtered_masks:
            mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                mask_poly = np.zeros(image_cv.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask_poly, [cnt], 255)
                blurred = cv2.GaussianBlur(image_cv, (25, 25), 0)
                result_img[mask_poly == 255] = blurred[mask_poly == 255]

        # Encode blurred image to JPEG and base64
        _, buffer = cv2.imencode(".jpg", result_img)
        output_bytes = buffer.tobytes()
        output_b64 = base64.b64encode(output_bytes).decode("utf-8")

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "image/jpeg"
            },
            "body": output_b64,
            "isBase64Encoded": True
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
