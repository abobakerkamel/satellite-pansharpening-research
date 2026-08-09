from pathlib import Path
import importlib.util
import torch.nn as nn
import torch.nn.functional as F

def import_hat_class(path=None):
    path=Path(path or Path(__file__).with_name('hat_arch_standalone.py'))
    if not path.exists(): raise FileNotFoundError('Run prepare_hat_arch.py first.')
    spec=importlib.util.spec_from_file_location('hat_arch_standalone',path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m.HAT

class RefinementBlock(nn.Module):
    def __init__(self,c=64):
        super().__init__(); self.block=nn.Sequential(nn.Conv2d(c,c,3,1,1),nn.GELU(),nn.Conv2d(c,c,3,1,1))
    def forward(self,x): return x+self.block(x)

class HATPanFusion(nn.Module):
    def __init__(self,hat_class=None):
        super().__init__(); HAT=hat_class or import_hat_class()
        self.hat=HAT(upscale=4,in_chans=7,img_size=64,window_size=16,compress_ratio=3,squeeze_factor=30,conv_scale=0.01,overlap_ratio=0.5,img_range=1.0,depths=[6,6,6,6,6,6],embed_dim=180,num_heads=[6,6,6,6,6,6],mlp_ratio=2,upsampler='pixelshuffle',resi_connection='1conv',use_checkpoint=False,drop_path_rate=0.0)
        self.hat.conv_last=nn.Conv2d(64,6,3,1,1); self.refine_head=nn.Conv2d(13,64,3,1,1)
        self.refine_body=nn.Sequential(*[RefinementBlock(64) for _ in range(4)]); self.refine_tail=nn.Conv2d(64,6,3,1,1)
    def forward(self,lr_ms,pan_hr):
        up=F.interpolate(lr_ms,size=pan_hr.shape[-2:],mode='bicubic',align_corners=False)
        pan_lr=F.interpolate(pan_hr,size=lr_ms.shape[-2:],mode='area')
        coarse=up+self.hat(__import__('torch').cat([lr_ms,pan_lr],1))
        ref=self.refine_tail(self.refine_body(self.refine_head(__import__('torch').cat([coarse,up,pan_hr],1))))
        return (coarse+ref).clamp(0,1)
