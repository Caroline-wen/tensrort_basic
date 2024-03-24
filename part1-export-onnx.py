import torch
import torchvision.models as models

model = models.resnet50(weights='ResNet50_Weights.DEFAULT')

input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, input, "resnet50-1.onnx")

