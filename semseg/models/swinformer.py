import torch
from torch import Tensor
from torch import nn, Tensor
from torch.nn import functional as F
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from semseg.models.backbones.swin_transformer import SwinTransformer
from semseg.models.heads import UPerHead

swinformer_settings = {
    'B': [0.2, 128, [2, 2, 18, 2], [4, 8, 16, 32], 12, [128, 256, 512, 1024]],       # [dpr, dim, depths, num_head, windows, out_dims]
    'L': [0.2, 192, [2, 2, 18, 2], [6, 12, 24, 48], 12, [192, 384, 768, 1536]],
    'S': [0.3, 96, [2, 2, 18, 2], [3, 6, 12, 24], 7, [96, 192, 384, 768]],
    'T': [0.2, 96, [2, 2, 6, 2], [3, 6, 12, 24], 7, [96, 192, 384, 768]]
}

class Swinformer(nn.Module):
    def __init__(self, backbone_type='B', num_classes=19):
        super().__init__()
        back_param = swinformer_settings[backbone_type]
        self.backbone = SwinTransformer(drop_path_rate=back_param[0], embed_dim=back_param[1], 
                                        depths=back_param[2], num_heads=back_param[3], 
                                        window_size=back_param[4])
        self.decode_head = UPerHead(back_param[5], 512, num_classes=num_classes)
        # self.decode_head = UPerHead(back_param[5], 128, num_classes=num_classes)
        
    def forward(self, x: Tensor) -> Tensor:
        y = self.backbone(x)
        y = self.decode_head(y)
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)    # to original image shape
        
        return y
    
    
if __name__ == '__main__':
    model = Swinformer('T', num_classes=2)
    model.eval()
    x = torch.zeros(1, 3, 448, 448)
    y = model(x)
    print(y.shape)
    import numpy as np
    total_params = sum(p.numel() for p in model.parameters())
    print('Total number of parameters: %0.2f M'%(np.float32(total_params)/1000000))