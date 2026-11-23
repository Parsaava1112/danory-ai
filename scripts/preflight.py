import json
from pathlib import Path
from tokenizers import Tokenizer
root=Path(__file__).resolve().parents[1]
mc=json.loads((root/'configs/model.json').read_text());cfg=json.loads((root/'configs/pretrain.json').read_text())
tp=root/cfg['tokenizer_dir']/'tokenizer.json'
if not tp.exists():raise SystemExit(f'Missing tokenizer: {tp}. Run build_tokenizer.py first.')
t=Tokenizer.from_file(str(tp))
if t.get_vocab_size()!=mc['vocab_size']:raise SystemExit(f'Vocab mismatch: tokenizer={t.get_vocab_size()} model={mc["vocab_size"]}')
for s in ('<|unk|>','<|bos|>','<|eos|>','<|pad|>','<|system|>','<|user|>','<|assistant|>','<|tool|>','<|end|>'):
 if t.token_to_id(s) is None:raise SystemExit(f'Missing special token {s}')
for name in ('train_data','val_data'):
 p=root/cfg[name]
 if not p.exists():raise SystemExit(f'Missing {name}: {p}')
print('Preflight passed: tokenizer, special tokens, vocab and pretraining data.')
