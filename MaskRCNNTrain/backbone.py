import torch
import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

def get_backbone():
    # Use a ResNet50 with FPN (Feature Pyramid Network)
    backbone = resnet_fpn_backbone('resnet50', pretrained=True)
    return backbone
