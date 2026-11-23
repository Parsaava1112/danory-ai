import json,torch
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from danory.model import DanoryGPT
root=Path(__file__).resolve().parents[1];cfg=json.loads((root/'configs/model.json').read_text())
m=DanoryGPT(cfg);x=torch.randint(0,cfg['vocab_size'],(1,12));y=x.clone();logits,loss=m(x,y)
assert logits.shape==(1,12,cfg['vocab_size']) and torch.isfinite(loss)
loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
assert m.token_emb.weight.data_ptr()==m.lm_head.weight.data_ptr()
print('Model forward/backward passed; parameters:',m.parameter_count())
