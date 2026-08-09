import argparse,sys,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from models.hat.model import HATPanFusion
from models.common.inference_utils import full_scene_tiled_inference
def main():
    p=argparse.ArgumentParser();p.add_argument('--ms',required=True);p.add_argument('--pan',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--stats',default=str(ROOT/'weights'/'train_normalization_stats.json'));a=p.parse_args();d=torch.device('cuda' if torch.cuda.is_available() else 'cpu');m=HATPanFusion().to(d);c=torch.load(a.checkpoint,map_location=d,weights_only=False);m.load_state_dict(c['model_state_dict'],strict=True);print(full_scene_tiled_inference(m,a.ms,a.pan,a.stats,a.output,tile_lr=128,device=d))
if __name__=='__main__':main()
