"""Danory GPT v8 decoder-only Transformer: GQA, RoPE, RMSNorm, SwiGLU."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__(); self.weight=nn.Parameter(torch.ones(dim)); self.eps=eps
    def forward(self,x):
        scale=torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+self.eps)
        return x*scale.to(x.dtype)*self.weight

def rotate_half(x):
    a,b=x.chunk(2,dim=-1); return torch.cat((-b,a),dim=-1)

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, theta=10000.0):
        super().__init__()
        if head_dim%2: raise ValueError("RoPE requires even head dimension")
        inv=1.0/(theta**(torch.arange(0,head_dim,2,dtype=torch.float32)/head_dim))
        self.register_buffer("inv_freq",inv,persistent=False)
    def forward(self,x,t):
        pos=torch.arange(t,device=x.device,dtype=self.inv_freq.dtype)
        f=torch.outer(pos,self.inv_freq); emb=torch.cat((f,f),dim=-1)
        return emb.cos().to(x.dtype)[None,None],emb.sin().to(x.dtype)[None,None]

class CausalSelfAttention(nn.Module):
    def __init__(self,cfg):
        super().__init__(); d=int(cfg['n_embd']); self.n_head=int(cfg['n_head']); self.n_kv=int(cfg['n_kv_head'])
        if d%self.n_head or self.n_head%self.n_kv or (d//self.n_head)%2: raise ValueError("Invalid GQA/RoPE dimensions")
        self.hd=d//self.n_head; kvd=self.n_kv*self.hd; bias=bool(cfg.get('bias',False))
        self.q=nn.Linear(d,d,bias=bias); self.k=nn.Linear(d,kvd,bias=bias); self.v=nn.Linear(d,kvd,bias=bias); self.o=nn.Linear(d,d,bias=bias)
        self.rope=RotaryEmbedding(self.hd,float(cfg.get('rope_theta',10000)))
        self.dropout=float(cfg.get('attention_dropout',cfg.get('dropout',0)))
    def forward(self,x):
        b,t,d=x.shape
        q=self.q(x).view(b,t,self.n_head,self.hd).transpose(1,2)
        k=self.k(x).view(b,t,self.n_kv,self.hd).transpose(1,2)
        v=self.v(x).view(b,t,self.n_kv,self.hd).transpose(1,2)
        cos,sin=self.rope(q,t); q=q*cos+rotate_half(q)*sin; k=k*cos+rotate_half(k)*sin
        if self.n_kv!=self.n_head:
            r=self.n_head//self.n_kv; k=k.repeat_interleave(r,1); v=v.repeat_interleave(r,1)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=self.dropout if self.training else 0.)
        return self.o(y.transpose(1,2).contiguous().view(b,t,d))

class SwiGLU(nn.Module):
    def __init__(self,d,h,bias=False):
        super().__init__(); self.gate=nn.Linear(d,h,bias=bias); self.up=nn.Linear(d,h,bias=bias); self.down=nn.Linear(h,d,bias=bias)
    def forward(self,x): return self.down(F.silu(self.gate(x))*self.up(x))

class Block(nn.Module):
    def __init__(self,cfg):
        super().__init__(); d=int(cfg['n_embd']); eps=float(cfg.get('rms_norm_eps',1e-6))
        self.n1=RMSNorm(d,eps); self.attn=CausalSelfAttention(cfg); self.n2=RMSNorm(d,eps)
        self.mlp=SwiGLU(d,int(cfg['ffn_dim']),bool(cfg.get('bias',False))); self.drop=nn.Dropout(float(cfg.get('dropout',0)))
    def forward(self,x):
        x=x+self.drop(self.attn(self.n1(x)))
        return x+self.drop(self.mlp(self.n2(x)))

class DanoryGPT(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=dict(cfg); self.validate_config(cfg)
        d=int(cfg['n_embd']); v=int(cfg['vocab_size']); init=float(cfg.get('initializer_range',.02))
        self.token_emb=nn.Embedding(v,d); self.emb_drop=nn.Dropout(float(cfg.get('dropout',0)))
        self.blocks=nn.ModuleList([Block(cfg) for _ in range(int(cfg['n_layer']))]); self.norm=RMSNorm(d,float(cfg.get('rms_norm_eps',1e-6)))
        self.lm_head=nn.Linear(d,v,bias=False)
        if cfg.get('tie_weights',True): self.lm_head.weight=self.token_emb.weight
        self.apply(self._init); nn.init.normal_(self.token_emb.weight,0,init)
        scale=init/math.sqrt(2*int(cfg['n_layer']))
        for b in self.blocks: nn.init.normal_(b.attn.o.weight,0,scale); nn.init.normal_(b.mlp.down.weight,0,scale)
    @staticmethod
    def validate_config(c):
        keys=('vocab_size','block_size','n_layer','n_head','n_kv_head','n_embd','ffn_dim')
        missing=[k for k in keys if k not in c]
        if missing: raise KeyError(f"Missing model config keys: {missing}")
        if any(int(c[k])<=0 for k in keys): raise ValueError("Model dimensions must be positive")
        if int(c['n_embd'])%int(c['n_head']) or int(c['n_head'])%int(c['n_kv_head']) or (int(c['n_embd'])//int(c['n_head']))%2: raise ValueError("Invalid attention dimensions")
        if not 0<=float(c.get('dropout',0))<1: raise ValueError("dropout must be in [0,1)")
    @staticmethod
    def _init(m):
        if isinstance(m,nn.Linear): nn.init.normal_(m.weight,0,.02); nn.init.zeros_(m.bias) if m.bias is not None else None
        elif isinstance(m,nn.Embedding): nn.init.normal_(m.weight,0,.02)
    def parameter_count(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)
    def forward(self,idx,targets=None,loss_mask=None):
        if idx.ndim!=2 or idx.size(1)<1: raise ValueError("idx must be [batch, non-empty sequence]")
        if idx.size(1)>int(self.cfg['block_size']): raise ValueError("Input exceeds block_size")
        if idx.min().item()<0 or idx.max().item()>=int(self.cfg['vocab_size']): raise ValueError("Token ID outside vocabulary")
        x=self.emb_drop(self.token_emb(idx))
        for block in self.blocks: x=block(x)
        logits=self.lm_head(self.norm(x)); loss=None
        if targets is not None:
            if targets.shape!=idx.shape: raise ValueError("targets shape mismatch")
            raw=F.cross_entropy(logits.float().reshape(-1,logits.size(-1)),targets.reshape(-1),reduction='none',ignore_index=-100).view_as(targets)
            mask=targets.ne(-100).to(raw.dtype)
            if loss_mask is not None:
                if loss_mask.shape!=targets.shape: raise ValueError("loss_mask shape mismatch")
                mask=mask*loss_mask.to(raw.device,dtype=raw.dtype)
            denom=mask.sum()
            if denom.item()==0: raise ValueError("Batch contains no supervised tokens")
            loss=(raw*mask).sum()/denom
        return logits,loss
    @staticmethod
    def _top_k(x,k):
        if not k or k<=0:return x
        k=min(int(k),x.size(-1)); cutoff=torch.topk(x,k,dim=-1).values[...,-1,None]; return x.masked_fill(x<cutoff,-torch.inf)
    @staticmethod
    def _top_p(x,p):
        if p>=1:return x
        vals,ids=torch.sort(x,descending=True,dim=-1); remove=F.softmax(vals,dim=-1).cumsum(-1)>p
        remove[...,1:]=remove[...,:-1].clone(); remove[...,0]=False
        mask=torch.zeros_like(remove).scatter(-1,ids,remove); return x.masked_fill(mask,-torch.inf)
    @torch.inference_mode()
    def generate(self,idx,max_new_tokens=128,temperature=.8,top_k=50,top_p=.92,repetition_penalty=1.08,eos_token_id=None,forbidden_token_ids=()):
        self.eval()
        if idx.ndim!=2 or idx.size(1)==0: raise ValueError("Prompt must be non-empty [batch, sequence]")
        if temperature<=0 or not 0<top_p<=1 or repetition_penalty<1 or max_new_tokens<0: raise ValueError("Invalid generation settings")
        idx=idx.long(); finished=torch.zeros(idx.size(0),dtype=torch.bool,device=idx.device)
        for step in range(int(max_new_tokens)):
            logits=self(idx[:,-int(self.cfg['block_size']):])[0][:,-1,:].float()
            if repetition_penalty>1:
                for b in range(idx.size(0)):
                    recent=torch.unique(idx[b,-128:]); values=logits[b,recent]
                    logits[b,recent]=torch.where(values<0,values*repetition_penalty,values/repetition_penalty)
            for tid in forbidden_token_ids:
                if tid is not None: logits[:,int(tid)]=-torch.inf
            if eos_token_id is not None and step<2: logits[:,int(eos_token_id)]=-torch.inf
            logits=self._top_p(self._top_k(logits/temperature,top_k),top_p)
            nxt=torch.multinomial(F.softmax(logits,dim=-1),1)
            if eos_token_id is not None:
                nxt=torch.where(finished[:,None],torch.full_like(nxt,int(eos_token_id)),nxt); finished|=nxt[:,0].eq(int(eos_token_id))
            idx=torch.cat((idx,nxt),dim=1)
            if eos_token_id is not None and bool(finished.all()): break
        return idx
