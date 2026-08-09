import argparse,sys,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from models.pantiny.model import pantiny_small
from models.common.inference_utils import full_scene_tiled_inference
def load_model(checkpoint,device):
    m=pantiny_small(); c=torch.load(checkpoint,map_location=device,weights_only=False); s=c.get('model_state_dict',c); m.load_state_dict({k.removeprefix('module.'):v for k,v in s.items()},strict=True); return m
def main():
    p=argparse.ArgumentParser();p.add_argument('--ms',required=True);p.add_argument('--pan',required=True);p.add_argument('--output',required=True);p.add_argument('--checkpoint',default=str(Path(__file__).with_name('weights')/'best_pantiny_small_6band.pth'));p.add_argument('--stats',default=str(ROOT/'weights'/'train_normalization_stats.json'));a=p.parse_args();d=torch.device('cuda' if torch.cuda.is_available() else 'cpu');print(full_scene_tiled_inference(load_model(a.checkpoint,d),a.ms,a.pan,a.stats,a.output,tile_lr=160,device=d))
if __name__=='__main__':main()
