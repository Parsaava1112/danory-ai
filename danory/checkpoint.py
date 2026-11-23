import os,random,hashlib
from pathlib import Path
import numpy as np
import torch
FORMAT_VERSION=4

def tokenizer_fingerprint(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def rng_state():
    s={'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state()}
    if torch.cuda.is_available():s['cuda']=torch.cuda.get_rng_state_all()
    return s
def save_checkpoint(path,model,optimizer,step,config,best_val,tokenizer_hash,stage,data_fingerprint=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save({'format_version':FORMAT_VERSION,'model':model.state_dict(),'optimizer':optimizer.state_dict() if optimizer else None,
      'step':int(step),'stage':stage,'config':dict(config),'best_val':float(best_val),'tokenizer_hash':tokenizer_hash,
      'model_config':dict(model.cfg),'data_fingerprint':data_fingerprint,'rng_state':rng_state(),'torch_version':torch.__version__},tmp)
    os.replace(tmp,path)
def read_checkpoint(path):return torch.load(Path(path),map_location='cpu',weights_only=False)
def load_checkpoint(path,model,optimizer=None,map_location='cpu',expected_tokenizer_hash=None,expected_stage=None,expected_data_fingerprint=None):
    ck=torch.load(Path(path),map_location=map_location,weights_only=False)
    if ck.get('format_version')!=FORMAT_VERSION:raise ValueError(f"Checkpoint format {ck.get('format_version')} incompatible with {FORMAT_VERSION}")
    if ck.get('model_config')!=model.cfg:raise ValueError('Checkpoint model architecture differs from current config')
    if expected_tokenizer_hash and ck.get('tokenizer_hash')!=expected_tokenizer_hash:raise ValueError('Tokenizer fingerprint mismatch')
    if expected_stage and ck.get('stage')!=expected_stage:raise ValueError('Checkpoint stage mismatch')
    if expected_data_fingerprint and ck.get('data_fingerprint')!=expected_data_fingerprint:raise ValueError('Dataset manifest changed since checkpoint; do not resume on a different dataset mix.')
    model.load_state_dict(ck['model'],strict=True)
    if optimizer is not None and ck.get('optimizer'):optimizer.load_state_dict(ck['optimizer'])
    return ck
def restore_rng(s):
    if not s:return
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'])
    if torch.cuda.is_available() and s.get('cuda'):torch.cuda.set_rng_state_all(s['cuda'])
