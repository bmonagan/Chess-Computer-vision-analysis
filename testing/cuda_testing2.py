import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
import os
print(os.getcwd())
print(os.environ.get("CUDA_VISIBLE_DEVICES"))