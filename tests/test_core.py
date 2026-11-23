import sys,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from danory.model import DanoryGPT

class CoreTests(unittest.TestCase):
    def test_architecture_and_forward(self):
        import json
        cfg=json.loads((ROOT/'configs/model.json').read_text(encoding='utf-8'))
        model=DanoryGPT(cfg)
        self.assertGreater(model.parameter_count(),95_000_000)
        self.assertLess(model.parameter_count(),105_000_000)
        x=torch.randint(0,cfg['vocab_size'],(1,8));logits,loss=model(x,x)
        self.assertEqual(tuple(logits.shape),(1,8,cfg['vocab_size']))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
    def test_masked_loss_ignores_padding(self):
        import json
        cfg=json.loads((ROOT/'configs/model.json').read_text(encoding='utf-8'))
        cfg.update(n_layer=1,n_embd=64,n_head=4,n_kv_head=2,ffn_dim=128,vocab_size=128,block_size=8)
        m=DanoryGPT(cfg);x=torch.randint(0,128,(1,8));y=x.clone();mask=torch.zeros_like(x,dtype=torch.float32)
        with self.assertRaises(ValueError):m(x,y,mask)


class FakeEncoding:
    def __init__(self,ids):self.ids=ids
class FakeTokenizer:
    def __init__(self):
        self.special={'<|unk|>':0,'<|bos|>':1,'<|eos|>':2,'<|pad|>':3,'<|system|>':4,'<|user|>':5,'<|assistant|>':6,'<|tool|>':7,'<|end|>':8}
    def token_to_id(self,t):return self.special.get(t)
    def encode(self,text,add_special_tokens=False):return FakeEncoding([10+(ord(c)%20) for c in text])

class DataTests(unittest.TestCase):
    def test_sft_masks_only_assistant_targets(self):
        import tempfile,json
        from danory.data import JsonlTokenDataset
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'sft.jsonl';path.write_text(json.dumps({'messages':[{'role':'user','content':'سلام'}, {'role':'assistant','content':'پاسخ'}]},ensure_ascii=False)+'\n',encoding='utf-8')
            ds=JsonlTokenDataset(path,FakeTokenizer(),64,stage='sft',shuffle_buffer=0)
            x,y,mask=ds.sft_item(json.loads(path.read_text(encoding='utf-8')))
            self.assertGreater(int(mask.sum()),0)
            self.assertTrue(torch.all(y[mask.eq(0)]==-100))
            self.assertTrue(torch.all(y[mask.eq(1)]!=-100))



class UtilityTests(unittest.TestCase):
    def test_request_analyzer(self):
        from danory.analyzer import analyze_request
        self.assertEqual(analyze_request('این متن را خلاصه کن')['intent'],'summarization')
        self.assertEqual(analyze_request('تفاوت این دو روش چیست؟')['intent'],'comparison')
    def test_checkpoint_roundtrip(self):
        import tempfile
        from danory.checkpoint import save_checkpoint,load_checkpoint
        cfg={'vocab_size':64,'block_size':8,'n_layer':1,'n_head':4,'n_kv_head':2,'n_embd':32,'ffn_dim':64,'dropout':0.0,'tie_weights':True}
        model=DanoryGPT(cfg);opt=torch.optim.AdamW(model.parameters())
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'last.pt'
            save_checkpoint(path,model,opt,7,{'max_steps':10},1.23,'token-hash','pretrain','data-hash')
            other=DanoryGPT(cfg);other_opt=torch.optim.AdamW(other.parameters())
            ck=load_checkpoint(path,other,other_opt,expected_tokenizer_hash='token-hash',expected_stage='pretrain',expected_data_fingerprint='data-hash')
            self.assertEqual(ck['step'],7)


if __name__=='__main__':unittest.main()
