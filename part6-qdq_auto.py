import torch
import torchvision
from pytorch_quantization import tensor_quant
from pytorch_quantization import quant_modules
from pytorch_quantization import nn as quant_nn

quant_modules.initialize()
model = torchvision.models.resnet50()
model.cuda()

inputs = torch.randn(1, 3, 224, 224, device = 'cuda')
quant_nn.TensorQuantizer.use_fb_fake_quant = True
torch.onnx.export(model, inputs, 'quant_resnet50_1.onnx', opset_version = 13)