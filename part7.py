import torch
import torchvision
from pytorch_quantization import tensor_quant
from pytorch_quantization import quant_modules
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.nn.modules import _utils as quant_in_utils
from pytorch_quantization import calib
from typing import List, Callable, Union, Dict
from pytorch_quantization.tensor_quant import QuantDescriptor

class QuantMultiAdd(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._input_quantizer = quant_nn.TensorQuantizer(QuantDescriptor(num_bits=8, calib_method="histogram"))
    def forward(self, x, y, z):
        return self._input_quantizer(x) + self._input_quantizer(y)+self._input_quantizer(z)

# quant_modules.initialize()
model = QuantMultiAdd()
model.cuda()
# disable_quantization(model.conv1).apply()
# quantizer_state(model)
inputs_a = torch.randn(1, 3, 224, 224, device='cuda')
inputs_b = torch.randn(1, 3, 224, 224, device='cuda')
inputs_c = torch.randn(1, 3, 224, 224, device='cuda')
quant_nn.TensorQuantizer.use_fb_fake_quant = True
torch.onnx.export(model, (inputs_a, inputs_b, inputs_c), 'quantMultiAdd.onnx', opset_version=13)
