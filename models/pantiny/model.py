from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass(frozen=True)
class PanTinyConfig:
    ms_bands:int=6; scale:int=4; dim:int=24; depth:int=4; heads:int=3; ffn_expansion:float=2.0

class LayerNorm2d(nn.Module):
    def __init__(self,c):
        super().__init__(); self.weight=nn.Parameter(torch.ones(c)); self.bias=nn.Parameter(torch.zeros(c))
    def forward(self,x):
        m=x.mean(1,keepdim=True); v=x.var(1,keepdim=True,unbiased=False)
        x=(x-m)/torch.sqrt(v+1e-5)
        return x*self.weight[None,:,None,None]+self.bias[None,:,None,None]

class ChannelAttention(nn.Module):
    def __init__(self,dim,heads):
        super().__init__(); self.heads=heads; self.temperature=nn.Parameter(torch.ones(heads,1,1))
        self.qkv=nn.Conv2d(dim,dim*3,1,bias=False)
        self.qkv_dwconv=nn.Conv2d(dim*3,dim*3,3,1,1,groups=dim*3,bias=False)
        self.project_out=nn.Conv2d(dim,dim,1,bias=False)
    def forward(self,x):
        b,c,h,w=x.shape; d=c//self.heads
        q,k,v=self.qkv_dwconv(self.qkv(x)).chunk(3,1)
        q=F.normalize(q.reshape(b,self.heads,d,h*w),dim=-1); k=F.normalize(k.reshape(b,self.heads,d,h*w),dim=-1)
        v=v.reshape(b,self.heads,d,h*w); a=((q@k.transpose(-2,-1))*self.temperature).softmax(-1)
        return self.project_out((a@v).reshape(b,c,h,w))

class GatedDConvFeedForward(nn.Module):
    def __init__(self,dim,exp):
        super().__init__(); hidden=int(dim*exp)
        self.project_in=nn.Conv2d(dim,hidden*2,1,bias=False)
        self.depthwise=nn.Conv2d(hidden*2,hidden*2,3,1,1,groups=hidden*2,bias=False)
        self.project_out=nn.Conv2d(hidden,dim,1,bias=False)
    def forward(self,x):
        a,b=self.depthwise(self.project_in(x)).chunk(2,1); return self.project_out(F.gelu(a)*b)

class TransformerBlock(nn.Module):
    def __init__(self,dim,heads,exp):
        super().__init__(); self.norm1=LayerNorm2d(dim); self.attention=ChannelAttention(dim,heads); self.norm2=LayerNorm2d(dim); self.ffn=GatedDConvFeedForward(dim,exp)
    def forward(self,x):
        x=x+self.attention(self.norm1(x)); return x+self.ffn(self.norm2(x))

class EnhancedConv(nn.Module):
    def __init__(self,dim):
        super().__init__(); self.body=nn.Sequential(nn.Conv2d(dim,dim,3,padding=1,bias=False),nn.GELU(),nn.Conv2d(dim,dim,3,padding=1,bias=False))
    def forward(self,x): return x+self.body(x)

class PanTiny(nn.Module):
    def __init__(self,config=PanTinyConfig()):
        super().__init__(); self.config=config
        self.ms_encoder=nn.Sequential(nn.Conv2d(6,config.dim,3,padding=1,bias=False),nn.GELU())
        self.pan_projection=nn.Conv2d(1,config.dim,3,padding=1,bias=False)
        self.fusion=nn.Sequential(nn.Conv2d(config.dim*2,config.dim,3,padding=1,bias=False),nn.GELU())
        self.body=nn.Sequential(*[TransformerBlock(config.dim,config.heads,config.ffn_expansion) for _ in range(config.depth)])
        self.enhanced_conv=EnhancedConv(config.dim); self.output=nn.Conv2d(config.dim,6,3,padding=1,bias=True)
    def forward(self,lr_ms,pan):
        up=F.interpolate(lr_ms,size=pan.shape[-2:],mode='bicubic',align_corners=False)
        fused=self.fusion(torch.cat([self.ms_encoder(up),self.pan_projection(pan)],1))
        residual=self.output(self.enhanced_conv(fused+self.body(fused)))
        return (up+residual).clamp(0,1)

def pantiny_small(): return PanTiny(PanTinyConfig())
