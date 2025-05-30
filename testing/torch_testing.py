import torch
import torchvision
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch version:", torch.__version__)
print("torchvision version:", torchvision.__version__)
print("torchvision.ops.nms:", torchvision.ops.nms)