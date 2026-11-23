import argparse,json,unicodedata
from pathlib import Path
from tokenizers import Tokenizer,models,trainers,pre_tokenizers,normalizers,decoders
SPECIAL=['<|unk|>','<|bos|>','<|eos|>','<|pad|>','<|system|>','<|user|>','<|assistant|>','<|tool|>','<|end|>']
def stream(paths,max_chars):
    seen=0
    for path in paths:
        with path.open(encoding='utf-8',errors='replace') as f:
            for line in f:
                try:row=json.loads(line)
                except ValueError:continue
                text=row.get('text','') if isinstance(row,dict) else ''
                if not isinstance(text,str):continue
                text=unicodedata.normalize('NFC',text).replace('ي','ی').replace('ك','ک').replace('ى','ی').replace('ـ','').strip()
                if len(text)<8:continue
                remain=max_chars-seen
                if remain<=0:return
                text=text[:remain];seen+=len(text);yield text

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',default='data/public/train.jsonl');p.add_argument('--out',default='data/tokenizer');p.add_argument('--vocab-size',type=int,default=24000);p.add_argument('--max-chars',type=int,default=300_000_000);a=p.parse_args()
    paths=[Path(x) for x in a.input.split(',')]
    for f in paths:
        if not f.is_file():raise FileNotFoundError(f)
    tok=Tokenizer(models.BPE(unk_token='<|unk|>'))
    tok.normalizer=normalizers.Sequence([normalizers.NFC(),normalizers.Replace('ي','ی'),normalizers.Replace('ك','ک'),normalizers.Replace('ى','ی'),normalizers.Replace('ـ','')])
    tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False);tok.decoder=decoders.ByteLevel()
    trainer=trainers.BpeTrainer(vocab_size=a.vocab_size,min_frequency=2,special_tokens=SPECIAL,initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),show_progress=True)
    tok.train_from_iterator(stream(paths,a.max_chars),trainer=trainer)
    actual=tok.get_vocab_size();dest=Path(a.out);dest.mkdir(parents=True,exist_ok=True);tok.save(str(dest/'tokenizer.json'))
    report={'requested_vocab_size':a.vocab_size,'actual_vocab_size':actual,'max_chars':a.max_chars,'normalization':'NFC; Arabic Yeh/Kaf normalized; tatweel removed','special_tokens':{s:tok.token_to_id(s) for s in SPECIAL}}
    (dest/'tokenizer_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if actual!=a.vocab_size:raise SystemExit(f'Actual vocab ({actual}) differs from requested ({a.vocab_size}); update model config to exact value before training.')
if __name__=='__main__':main()
