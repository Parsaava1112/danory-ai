import argparse,json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from danory.model import DanoryGPT
from danory.chat import render_prompt
from danory.analyzer import analyze_request,response_guidance
from danory.checkpoint import load_checkpoint,tokenizer_fingerprint
from danory.trainer import resolve

def main():
    p=argparse.ArgumentParser(description='Danory GPT v8 text/chat generation')
    p.add_argument('--checkpoint',default='checkpoints/sft_best.pt');p.add_argument('--config',default='configs/pretrain.json')
    p.add_argument('--prompt',default=None);p.add_argument('--system',default='تو یک دستیار فارسی‌زبان دقیق، روشن و مفید هستی.')
    p.add_argument('--history',default=None,help='Optional UTF-8 JSON file containing prior role/content messages')
    p.add_argument('--tokens',type=int,default=160);p.add_argument('--temperature',type=float,default=.75);p.add_argument('--top-k',type=int,default=50);p.add_argument('--top-p',type=float,default=.92)
    p.add_argument('--greedy',action='store_true');a=p.parse_args()
    if a.prompt is None:p.error('--prompt is required')
    cfg=json.loads(resolve(a.config).read_text(encoding='utf-8'));mc=json.loads(resolve(cfg['model_config']).read_text(encoding='utf-8'))
    tp=resolve(cfg['tokenizer_dir'])/'tokenizer.json';tok=Tokenizer.from_file(str(tp))
    if tok.get_vocab_size()!=mc['vocab_size']:raise ValueError('Tokenizer/model vocabulary mismatch')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(int(cfg.get('cpu_threads',2)))
    model=DanoryGPT(mc).to(device);ck=load_checkpoint(resolve(a.checkpoint),model,map_location=device,expected_tokenizer_hash=tokenizer_fingerprint(tp))
    analysis=analyze_request(a.prompt)
    system=a.system+'\nراهنمای این درخواست: '+response_guidance(analysis)
    messages=[{'role':'system','content':system}]
    if a.history:
        history=json.loads(resolve(a.history).read_text(encoding='utf-8'))
        if not isinstance(history,list):raise ValueError('--history must contain a JSON array of role/content messages')
        messages.extend(history)
    messages.append({'role':'user','content':a.prompt})
    ids=render_prompt(messages,tok)
    if len(ids)>mc['block_size']:ids=ids[-mc['block_size']:]
    x=torch.tensor([ids],dtype=torch.long,device=device)
    forbidden=[tok.token_to_id(t) for t in ('<|unk|>','<|pad|>','<|bos|>','<|eos|>','<|system|>','<|user|>','<|tool|>')]
    eos=tok.token_to_id('<|end|>')
    with torch.inference_mode():out=model.generate(x,a.tokens,1.0 if a.greedy else a.temperature,1 if a.greedy else a.top_k,1.0 if a.greedy else a.top_p,1.08,eos,forbidden)
    new=out[0,len(ids):].tolist();print(f'Danory GPT v8 | stage={ck.get("stage")} | step={ck["step"]} | params={model.parameter_count():,} | intent={analysis["intent"]}')
    print(tok.decode(new,skip_special_tokens=True).strip())
if __name__=='__main__':main()
