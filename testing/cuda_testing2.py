import torch
import torchvision
print(torch.__version__)        # Should show +cu118
print(torchvision.__version__)  # Should show +cu118 or similar
print(torchvision.ops.nms)      # Should NOT error
print(torch.cuda.is_available())# Should be True