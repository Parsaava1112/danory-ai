import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from danory.model import DanoryGPT
cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/model.json').read_text());m=DanoryGPT(cfg);n=m.parameter_count();print(f'{n:,} parameters ({n/1e6:.3f}M)')
if not 95_000_000<=n<=110_000_000:raise SystemExit('Model is outside the intended 95M-110M parameter band')
