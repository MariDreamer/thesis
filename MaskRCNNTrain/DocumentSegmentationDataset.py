import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pycocotools import mask as mask_utils
import torchvision.transforms.functional as F
from torchvision.ops import masks_to_boxes

class DocumentSegmentationDataset(Dataset):
    def __init__(self, image_dir, label_data, transforms=None):
        self.image_dir = image_dir
        self.label_data = label_data
        self.transforms = transforms
        self.image_infos = self.label_data["images"]
        self.annotations = self.label_data["annotations"]

    def __getitem__(self, idx):
        image_info = self.image_infos[idx]
        img_path = os.path.join(self.image_dir, image_info["file_name"])
        image = Image.open(img_path).convert("RGB")
        image_id = image_info["id"]

        # Dimensions
        height = image_info["height"]
        width = image_info["width"]

        # Get annotations for this image
        anns = [a for a in self.annotations if a["image_id"] == image_id]

        masks = []
        labels = []

        for ann in anns:
            segmentation = ann.get("segmentation", [])
            if not segmentation:
                continue  # Skip if no segmentation

            # Convert polygon(s) to RLE, then to binary mask
            rles = mask_utils.frPyObjects(segmentation, height, width)
            rle = mask_utils.merge(rles)
            mask = mask_utils.decode(rle)

            if mask.ndim == 3:
                mask = mask.any(axis=2)

            masks.append(torch.as_tensor(mask, dtype=torch.uint8))

            # Ensure label is an int
            labels.append(int(ann["category_id"]))

        if masks:
            masks_tensor = torch.stack(masks)
            boxes_tensor = masks_to_boxes(masks_tensor)
        else:
            masks_tensor = torch.zeros((0, height, width), dtype=torch.uint8)
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels = []

        target = {
            "boxes": boxes_tensor,
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": masks_tensor,
            "image_id": torch.tensor(image_id, dtype=torch.int64)
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target

    def __len__(self):
        return len(self.image_infos)
