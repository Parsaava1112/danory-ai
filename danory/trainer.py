import csv,json,math,random,time,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from .model import DanoryGPT
from .data import JsonlTokenDataset,collate
from .checkpoint import save_checkpoint,load_checkpoint,read_checkpoint,restore_rng,tokenizer_fingerprint

ROOT=Path(__file__).resolve().parents[1]
def resolve(p):
    p=Path(p); return p if p.is_absolute() else ROOT/p

def load_json(p):return json.loads(resolve(p).read_text(encoding='utf-8'))
def fingerprint_files(paths):
    h=hashlib.sha256()
    for path in paths:
        path=Path(path)
        if not path.is_file():continue
        h.update(path.name.encode('utf-8'));h.update(str(path.stat().st_size).encode('ascii'))
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def lr_at(step,cfg):
    warm=max(0,int(cfg.get('warmup_steps',0))); total=int(cfg['max_steps'])
    peak=float(cfg['learning_rate']); floor=float(cfg.get('min_lr',peak*.1))
    if warm and step<warm:return peak*(step+1)/warm
    if total<=warm:return floor
    progress=min(1.,max(0.,(step-warm)/(total-warm)))
    return floor+(peak-floor)*.5*(1+math.cos(math.pi*progress))

@torch.no_grad()
def evaluate(model,loader,device,max_batches):
    model.eval(); losses=[]
    for i,batch in enumerate(loader):
        if max_batches and i>=max_batches:break
        x,y,m=(v.to(device) for v in batch)
        _,loss=model(x,y,m)
        if loss is not None and torch.isfinite(loss):losses.append(float(loss))
    model.train()
    return float(np.mean(losses)) if losses else float('nan')

def make_loader(path,tok,mc,cfg,stage,shuffle):
    ds=JsonlTokenDataset(resolve(path),tok,mc['block_size'],stage=stage,
       shuffle_buffer=int(cfg.get('shuffle_buffer',256)) if shuffle else 0,seed=int(cfg.get('seed',1337)))
    return DataLoader(ds,batch_size=int(cfg.get('batch_size',1)),num_workers=0,collate_fn=collate)

def run(config_path,stage='pretrain',resume=None,init_from=None,steps_this_run=None):
    cfg=load_json(config_path)
    if stage not in ('pretrain','sft'):raise ValueError('stage must be pretrain or sft')
    mc=load_json(cfg['model_config']);tok_path=resolve(cfg['tokenizer_dir'])/'tokenizer.json'
    if not tok_path.is_file():raise FileNotFoundError(f'Missing tokenizer: {tok_path}')
    tok=Tokenizer.from_file(str(tok_path))
    if tok.get_vocab_size()!=int(mc['vocab_size']):raise ValueError(f"Tokenizer vocab {tok.get_vocab_size()} != model vocab {mc['vocab_size']}")
    for token in ('<|unk|>','<|bos|>','<|eos|>','<|pad|>','<|system|>','<|user|>','<|assistant|>','<|end|>'):
        if tok.token_to_id(token) is None:raise ValueError(f'Tokenizer missing required token {token}')
    t_hash=tokenizer_fingerprint(tok_path)
    seed_all(int(cfg.get('seed',1337)))
    torch.set_num_threads(max(1,int(cfg.get('cpu_threads',2))))
    requested=str(cfg.get('device','auto'));device=torch.device('cuda' if requested=='auto' and torch.cuda.is_available() else ('cpu' if requested=='auto' else requested))
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    model=DanoryGPT(mc).to(device)
    print(f'Danory GPT v8 | stage={stage} | device={device} | parameters={model.parameter_count():,}',flush=True)
    data_key='train_data' if stage=='pretrain' else 'sft_data'
    train_path=cfg.get(data_key)
    if not train_path:raise ValueError(f'Config missing {data_key}')
    manifest=resolve('data/public/data_manifest.json')
    fingerprint_paths=[manifest,resolve(train_path)]
    if cfg.get('val_data') and resolve(cfg['val_data']).is_file():fingerprint_paths.append(resolve(cfg['val_data']))
    data_fingerprint=fingerprint_files(fingerprint_paths)
    train_loader=make_loader(train_path,tok,mc,cfg,stage,True)
    val_path=cfg.get('val_data')
    val_loader=make_loader(val_path,tok,mc,{**cfg,'shuffle_buffer':0},stage,False) if val_path and resolve(val_path).is_file() else None
    decay_params=[];no_decay_params=[]
    for name,param in model.named_parameters():
        if not param.requires_grad:continue
        if param.ndim>=2 and 'norm' not in name.lower() and 'token_emb' not in name:
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    groups=[{'params':decay_params,'weight_decay':float(cfg.get('weight_decay',.1))},
            {'params':no_decay_params,'weight_decay':0.0}]
    opt=torch.optim.AdamW(groups,lr=float(cfg['learning_rate']),betas=tuple(cfg.get('betas',[.9,.95])),eps=float(cfg.get('adam_eps',1e-8)))
    start=0;best=float('inf')
    if init_from:
        source=read_checkpoint(resolve(init_from))
        if source.get('format_version')!=4:raise ValueError('init-from checkpoint format is incompatible')
        if source.get('model_config')!=model.cfg:raise ValueError('init-from architecture mismatch; v7 weights cannot be loaded into v8')
        if source.get('tokenizer_hash')!=t_hash:raise ValueError('init-from tokenizer mismatch')
        model.load_state_dict(source['model'],strict=True)
        print(f"Initialized model weights from {init_from}; optimizer starts fresh.")
    if resume:
        rp=resolve(resume); meta=read_checkpoint(rp)
        if int(meta.get('config',{}).get('max_steps',-1))!=int(cfg['max_steps']):raise ValueError('Total max_steps changed. Keep the original schedule length when resuming.')
        ck=load_checkpoint(rp,model,opt,device,t_hash,stage,data_fingerprint)
        start=int(ck['step']);best=float(ck.get('best_val',float('inf')));restore_rng(ck.get('rng_state'))
        print(f'Resuming {stage} from step {start}/{cfg["max_steps"]}')
    out=resolve(cfg.get('out_dir','checkpoints'));logs=resolve(cfg.get('log_dir','logs'));out.mkdir(parents=True,exist_ok=True);logs.mkdir(parents=True,exist_ok=True)
    last=out/f'{stage}_last.pt';best_path=out/f'{stage}_best.pt';csv_path=logs/f'{stage}.csv'
    if not resume and csv_path.exists():csv_path.unlink()
    if not csv_path.exists():
        with csv_path.open('w',newline='',encoding='utf-8') as f:csv.writer(f).writerow(['step','loss','lr','grad_norm','val_loss','elapsed_sec'])
    total=int(cfg['max_steps']);accum=int(cfg.get('grad_accum',1))
    if total<1 or accum<1:raise ValueError('max_steps and grad_accum must be positive')
    if steps_this_run is not None and int(steps_this_run)<1:raise ValueError('steps_this_run must be positive')
    end=min(total,start+int(steps_this_run)) if steps_this_run is not None else total
    if start>=total:raise ValueError(f'Checkpoint already at step {start}; configured total is {total}')
    iterator=iter(train_loader);model.train();started=time.time();opt.zero_grad(set_to_none=True)
    for step in range(start,end):
        loss_sum=0.;last_grad=0.
        for _ in range(accum):
            try:batch=next(iterator)
            except StopIteration:
                iterator=iter(train_loader)
                try:batch=next(iterator)
                except StopIteration:raise RuntimeError('Training dataset yielded no usable samples')
            x,y,m=(v.to(device) for v in batch)
            _,loss=model(x,y,m)
            if loss is None or not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
            (loss/accum).backward();loss_sum+=float(loss.detach())
        last_grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg.get('grad_clip',1.0))))
        lr=lr_at(step,cfg)
        for group in opt.param_groups:group['lr']=lr
        opt.step();opt.zero_grad(set_to_none=True);done=step+1;val=float('nan')
        eval_every=max(1,int(cfg.get('eval_interval',100)))
        if val_loader is not None and (done%eval_every==0 or done==end or done==total):
            val=evaluate(model,val_loader,device,int(cfg.get('eval_batches',30)))
            if math.isfinite(val) and val<best:
                best=val;save_checkpoint(best_path,model,opt,done,cfg,best,t_hash,stage,data_fingerprint)
                print(f'New best checkpoint: {best_path} val_loss={best:.5f}',flush=True)
        save_every=max(1,int(cfg.get('save_interval',100)))
        if done%save_every==0 or done==end or done==total:
            save_checkpoint(last,model,opt,done,cfg,best,t_hash,stage,data_fingerprint)
        if done%max(1,int(cfg.get('log_interval',10)))==0 or done==end or done==total:
            elapsed=time.time()-started;msg=f'step={done}/{total} stage={stage} loss={loss_sum/accum:.4f} lr={lr:.3e} grad_norm={last_grad:.3f} elapsed={elapsed/60:.1f}min'
            if math.isfinite(val):msg+=f' val_loss={val:.5f}'
            print(msg,flush=True)
            with csv_path.open('a',newline='',encoding='utf-8') as f:csv.writer(f).writerow([done,loss_sum/accum,lr,last_grad,val,elapsed])
    if not best_path.exists():
        save_checkpoint(best_path,model,opt,end,cfg,best,t_hash,stage,data_fingerprint)
    print(f'Training segment finished at {end}/{total}. Last={last}; Best={best_path}; best_val={best}',flush=True)
