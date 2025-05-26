from torchvision.models.detection import MaskRCNN
import backbone

def build_instance_seg_model(num_classes):
    backbone_model = backbone.get_backbone()
    
    model = MaskRCNN(backbone_model, num_classes=num_classes)
    
    return model
