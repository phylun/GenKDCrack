import torch
from torch import Tensor
from torch import nn, Tensor
from torch.nn import functional as F
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from semseg.models.backbones import ConvNeXt
from semseg.models.heads import UPerHead


class Convnext(nn.Module):
    """
    Notes::::: This implementation has larger params and FLOPs than the results reported in the paper.
    Will update the code and weights if the original author releases the full code.
    """
    # def __init__(self, backbone: str = 'B', num_classes: int = 19) -> None:
    def __init__(self, backbone_type='B', num_classes=19) -> None:
        super().__init__()
        # super().__init__(backbone, num_classes)
        self.backbone = ConvNeXt(backbone_type)
        self.decode_head = UPerHead(self.backbone.channels, 128, num_classes=num_classes)        

    def forward(self, x: Tensor) -> Tensor:
        y = self.backbone(x)
        y = self.decode_head(y)   # 4x reduction in image size
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)    # to original image shape
        return y


if __name__ == '__main__':
    model = Convnext('T', num_classes=2)
    model.eval()
    x = torch.zeros(1, 3, 448, 448)
    y = model(x)
    print(y.shape)
    import numpy as np
    total_params = sum(p.numel() for p in model.parameters())
    print('Total number of parameters: %0.2f M'%(np.float32(total_params)/1000000))