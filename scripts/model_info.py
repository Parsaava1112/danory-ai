import json,argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from danory.model import DanoryGPT
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/model.json');a=p.parse_args()
c=json.loads(Path(a.config).read_text(encoding='utf-8'));m=DanoryGPT(c);n=m.parameter_count()
print(f'Parameters: {n:,} ({n/1e6:.3f}M)');print(f'FP32 weights: {n*4/1024**2:.1f} MiB');print('Tied embeddings:',m.token_emb.weight.data_ptr()==m.lm_head.weight.data_ptr())
