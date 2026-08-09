from pathlib import Path
import subprocess
HERE=Path(__file__).resolve().parent; REPO=HERE/'_HAT_source'; OUT=HERE/'hat_arch_standalone.py'
if not REPO.exists(): subprocess.run(['git','clone','https://github.com/XPixelGroup/HAT.git',str(REPO)],check=True)
source=(REPO/'hat'/'archs'/'hat_arch.py').read_text(encoding='utf-8')
source=source.replace('from basicsr.utils.registry import ARCH_REGISTRY',"""class _SimpleRegistry:\n    def register(self):\n        def decorator(obj): return obj\n        return decorator\nARCH_REGISTRY = _SimpleRegistry()""")
source=source.replace('from basicsr.archs.arch_util import to_2tuple, trunc_normal_','from timm.layers import to_2tuple, trunc_normal_')
OUT.write_text(source,encoding='utf-8'); print('Created:',OUT)
print('Historical limitation: the original notebook did not pin a HAT git commit.')
