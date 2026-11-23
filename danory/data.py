import json,random
from pathlib import Path
import torch
from torch.utils.data import IterableDataset
from .chat import encode_chat

class JsonlTokenDataset(IterableDataset):
    def __init__(self,path,tokenizer,block_size,stage='pretrain',shuffle_buffer=256,seed=1337):
        super().__init__(); self.path=Path(path); self.tok=tokenizer; self.block=int(block_size); self.stage=stage
        self.buf=max(0,int(shuffle_buffer)); self.seed=int(seed)
        if not self.path.is_file(): raise FileNotFoundError(f'Dataset file not found: {self.path}')
        if stage not in ('pretrain','sft'): raise ValueError(f'Unknown stage: {stage}')
    def rows(self):
        with self.path.open(encoding='utf-8',errors='replace') as f:
            for line in f:
                try: row=json.loads(line)
                except (ValueError,TypeError): continue
                if isinstance(row,dict): yield row
    def shuffled(self):
        if self.buf<=1: yield from self.rows(); return
        rng=random.Random(self.seed); buf=[]
        for row in self.rows():
            buf.append(row)
            if len(buf)>=self.buf: yield buf.pop(rng.randrange(len(buf)))
        rng.shuffle(buf); yield from buf
    def pretrain_tokens(self,row):
        text=row.get('text','')
        if not isinstance(text,str) or not text.strip():return []
        ids=self.tok.encode(text).ids; eos=self.tok.token_to_id('<|eos|>')
        if eos is not None:ids.append(eos)
        return ids
    def make_pretrain_block(self,ids):
        n=len(ids)-1
        x=torch.zeros(self.block,dtype=torch.long);y=torch.full((self.block,),-100,dtype=torch.long)
        x[:n]=torch.tensor(ids[:-1]);y[:n]=torch.tensor(ids[1:])
        return x,y,y.ne(-100).float()
    def sft_item(self,row):
        msgs=row.get('messages')
        if not isinstance(msgs,list):
            p=row.get('prompt',row.get('instruction',row.get('question',''))); a=row.get('output',row.get('response',row.get('answer','')))
            if not isinstance(p,str) or not isinstance(a,str) or not a.strip(): return None
            msgs=[{'role':'user','content':p},{'role':'assistant','content':a}]
        try: ids,mask=encode_chat(msgs,self.tok,self.block+1)
        except (ValueError,TypeError): return None
        if len(ids)<2 or not any(mask[1:]): return None
        pad=self.tok.token_to_id('<|pad|>') or 0; x=torch.full((self.block,),pad,dtype=torch.long)
        y=torch.full((self.block,),-100,dtype=torch.long); lm=torch.zeros(self.block)
        n=min(self.block,len(ids)-1); x[:n]=torch.tensor(ids[:n]); y[:n]=torch.tensor(ids[1:n+1])
        lm[:n]=torch.tensor(mask[1:n+1],dtype=torch.float32); y[lm.eq(0)]=-100
        return x,y,lm
    def __iter__(self):
        if self.stage=='pretrain':
            buffer=[]; width=self.block+1
            for row in self.shuffled():
                buffer.extend(self.pretrain_tokens(row))
                while len(buffer)>=width:
                    yield self.make_pretrain_block(buffer[:width])
                    del buffer[:self.block]
            if len(buffer)>=2:
                yield self.make_pretrain_block(buffer[:min(width,len(buffer))])
            return
        for row in self.shuffled():
            item=self.sft_item(row)
            if item is not None and item[1].ne(-100).any():yield item

def collate(batch): return tuple(torch.stack(x) for x in zip(*batch))
