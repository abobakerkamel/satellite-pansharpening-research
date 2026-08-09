from pathlib import Path
import json, numpy as np, rasterio, torch
from rasterio.enums import Resampling
from rasterio.windows import Window
from tqdm.auto import tqdm

def _stats(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def _norms(s):
    lo=np.asarray([x['p01'] for x in s['ms']],np.float32)[:,None,None]; hi=np.asarray([x['p99'] for x in s['ms']],np.float32)[:,None,None]
    plo=np.float32(s['pan']['p01']); phi=np.float32(s['pan']['p99'])
    return (lambda x:np.clip((x.astype(np.float32)-lo)/np.maximum(hi-lo,1e-6),0,1), lambda x:np.clip((x.astype(np.float32)-plo)/max(float(phi-plo),1e-6),0,1), lambda x:x.astype(np.float32)*(hi-lo)+lo)
def _pos(n,t,s):
    if n<=t:return [0]
    v=list(range(0,n-t+1,s));
    if v[-1]!=n-t:v.append(n-t)
    return v
def full_scene_tiled_inference(model,ms_path,pan_path,stats_path,output_path,scale=4,tile_lr=128,overlap_lr=32,device=None):
    device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu')); model=model.to(device).eval(); nms,npan,dms=_norms(_stats(stats_path))
    with rasterio.open(ms_path) as ms, rasterio.open(pan_path) as pan:
        if ms.count!=6: raise ValueError(f'Expected 6 MS bands, got {ms.count}')
        if pan.count!=1: raise ValueError(f'Expected 1 PAN band, got {pan.count}')
        if pan.width!=ms.width*scale or pan.height!=ms.height*scale: raise ValueError('PAN dimensions must be exactly ×4 of MS.')
        if ms.crs!=pan.crs: raise ValueError('MS/PAN CRS differ.')
        h,w=ms.height,ms.width; H,W=pan.height,pan.width; t=min(tile_lr,h,w); t=max(16,(t//16)*16); stride=t-overlap_lr; T=t*scale
        xs,ys=_pos(w,t,stride),_pos(h,t,stride); weight=np.maximum(np.outer(np.hanning(T),np.hanning(T)).astype(np.float32),1e-3)
        out=Path(output_path); out.parent.mkdir(parents=True,exist_ok=True); stem=out.with_suffix(''); sp=stem.with_name(stem.name+'_sum.dat'); wp=stem.with_name(stem.name+'_w.dat')
        sm=np.memmap(sp,'w+',np.float32,shape=(6,H,W)); sw=np.memmap(wp,'w+',np.float32,shape=(H,W)); sm[:]=0; sw[:]=0
        with torch.inference_mode():
            for y in tqdm(ys,desc='Inference rows'):
                for x in xs:
                    a=ms.read(window=Window(x,y,t,t)); b=pan.read(1,window=Window(x*scale,y*scale,T,T))[None]
                    at=torch.from_numpy(nms(a))[None].float().to(device); bt=torch.from_numpy(npan(b))[None].float().to(device)
                    with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'): pred=model(at,bt)[0].float().cpu().numpy()
                    yy,xx=y*scale,x*scale; sm[:,yy:yy+T,xx:xx+T]+=pred*weight[None]; sw[yy:yy+T,xx:xx+T]+=weight
        profile=pan.profile.copy()
        for k in ['blockxsize','blockysize','photometric','interleave']: profile.pop(k,None)
        profile.update(driver='GTiff',count=6,dtype='uint16',compress='deflate',predictor=2,tiled=True,blockxsize=256,blockysize=256,BIGTIFF='IF_SAFER')
        with rasterio.open(out,'w',**profile) as dst:
            B=512
            for y0 in tqdm(range(0,H,B),desc='Writing GeoTIFF'):
                bh=min(B,H-y0)
                for x0 in range(0,W,B):
                    bw=min(B,W-x0); num=np.asarray(sm[:,y0:y0+bh,x0:x0+bw]); den=np.asarray(sw[y0:y0+bh,x0:x0+bw]); pred=np.clip(num/np.maximum(den[None],1e-8),0,1); u16=np.clip(np.rint(dms(pred)),0,65535).astype(np.uint16); dst.write(u16,window=Window(x0,y0,bw,bh))
            dst.build_overviews([2,4,8,16],Resampling.average)
        del sm,sw
        for p in [sp,wp]:
            try:p.unlink()
            except OSError:pass
        return out
