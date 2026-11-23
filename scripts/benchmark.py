"""Small reproducible generation benchmark; quality scoring remains human-reviewed."""
import argparse,json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from danory.model import DanoryGPT
from danory.chat import render_prompt
from danory.checkpoint import load_checkpoint,tokenizer_fingerprint
from danory.trainer import resolve

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='checkpoints/sft_best.pt');p.add_argument('--config',default='configs/pretrain.json');p.add_argument('--prompts',default='tests/prompts_fa.json');p.add_argument('--out',default='reports/benchmark.json');a=p.parse_args()
 cfg=json.loads(resolve(a.config).read_text());mc=json.loads(resolve(cfg['model_config']).read_text());tp=resolve(cfg['tokenizer_dir'])/'tokenizer.json';tok=Tokenizer.from_file(str(tp))
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');m=DanoryGPT(mc).to(device);ck=load_checkpoint(resolve(a.checkpoint),m,map_location=device,expected_tokenizer_hash=tokenizer_fingerprint(tp))
 prompts=json.loads(resolve(a.prompts).read_text());results=[]
 for item in prompts:
  ids=render_prompt([{'role':'system','content':'به فارسی دقیق و روشن پاسخ بده.'},{'role':'user','content':item['prompt']}],tok)[-mc['block_size']:]
  x=torch.tensor([ids],device=device);out=m.generate(x,96,.7,40,.9,1.08,tok.token_to_id('<|end|>'),[tok.token_to_id(t) for t in ('<|unk|>','<|pad|>','<|bos|>','<|system|>','<|user|>','<|tool|>')])
  results.append({'category':item['category'],'prompt':item['prompt'],'completion':tok.decode(out[0,len(ids):].tolist(),skip_special_tokens=True)})
 dest=resolve(a.out);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps({'checkpoint_step':ck['step'],'results':results},ensure_ascii=False,indent=2),encoding='utf-8');print(f'Wrote {dest}')
if __name__=='__main__':main()
