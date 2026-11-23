"""Prepare bounded public Persian corpora. Never reads private/local user datasets."""
import argparse,hashlib,json,re,unicodedata
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import shutil

SOURCES={
 'naab':{'id':'SLPL/naab','license_note':'HF card currently labels MIT; corpus aggregates many upstream sources, review provenance and privacy notes.'},
 'fineweb2_hq_fa':{'id':'mohammadkrb/fineweb2-HQ-persian','license_note':'HF card labels ODC-BY; preserve attribution and verify terms for intended use.'},
 'cs_farsi':{'id':'Kamyar-zeinalipour/CS_Farsi','license_note':'HF card labels CC-BY-SA-4.0; attribution/share-alike obligations may apply.'},
 'farsinstruct':{'id':'ParsiAI/FarsInstruct','license_note':'HF card labels Apache-2.0; review attribution and source/task metadata.'},
 'persian_gk':{'id':'PersianML/persian-general-knowledge','license_note':'HF card labels CC-BY-4.0; preserve attribution.'},
}

def clean(text):
    s=unicodedata.normalize('NFC',str(text or '')).replace('ي','ی').replace('ك','ک').replace('ى','ی').replace('ـ','').replace('\u200f','').replace('\u200e','')
    s=re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]',' ',s);s=re.sub(r'\s+',' ',s).strip()
    return s

def is_persian_quality(s,min_ratio=.35):
    letters=[c for c in s if c.isalpha()]
    if len(letters)<12:return False
    fa=sum('\u0600'<=c<='\u06ff' for c in letters)
    return fa/max(1,len(letters))>=min_ratio

def export_text(dataset,out,limit,seen,source,min_chars=40,max_chars=12000,append=False):
    count=0;chars=0;out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('a' if append else 'w',encoding='utf-8') as f:
        for row in dataset:
            raw=row.get('text','') if isinstance(row,dict) else ''
            text=clean(raw)
            if len(text)<min_chars or not is_persian_quality(text):continue
            if len(text)>max_chars:text=text[:max_chars]
            digest=hashlib.sha256(text.encode()).hexdigest()
            if digest in seen:continue
            seen.add(digest)
            f.write(json.dumps({'text':text,'source':source},ensure_ascii=False)+'\n')
            count+=1;chars+=len(text)
            if count%10000==0:print(f'{source}: {count:,} accepted rows',flush=True)
            if limit and count>=limit:break
    print(f'{source}: wrote {count:,} rows / {chars:,} chars -> {out}',flush=True)
    return count

def to_messages(row):
    msgs=row.get('messages')
    if isinstance(msgs,list):
        out=[]
        for m in msgs:
            if not isinstance(m,dict):continue
            role=str(m.get('role',m.get('from',''))).lower()
            role={'human':'user','gpt':'assistant'}.get(role,role)
            content=m.get('content',m.get('value',''))
            if role in ('system','user','assistant') and isinstance(content,str) and content.strip():out.append({'role':role,'content':clean(content)})
        if any(m['role']=='user' for m in out) and any(m['role']=='assistant' for m in out):return out
    prompt=row.get('prompt',row.get('question',row.get('inputs','')))
    answer=row.get('assistant',row.get('answer',row.get('outputs',row.get('output',row.get('response','')))))
    if isinstance(prompt,str) and isinstance(answer,str) and len(answer.strip())>=8:
        return [{'role':'user','content':clean(prompt)},{'role':'assistant','content':clean(answer)}]
    return None

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--naab-train-rows',type=int,default=150000);p.add_argument('--naab-val-rows',type=int,default=12000)
    p.add_argument('--fineweb-rows',type=int,default=100000);p.add_argument('--sft-rows',type=int,default=20000)
    p.add_argument('--naab-train-shards',default='0');p.add_argument('--naab-test-shards',default='2')
    p.add_argument('--include-sft',action='store_true');p.add_argument('--out',default='data/public')
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    for v in (a.naab_train_rows,a.naab_val_rows,a.fineweb_rows,a.sft_rows):
        if v<0:raise ValueError('row limits must be non-negative')
    train=out/'train.jsonl';val=out/'validation.jsonl';seen=set()
    train_count=0
    shard_dir=out/'_naab_shards';shard_dir.mkdir(parents=True,exist_ok=True)
    def naab_lines(split,raw_indexes):
        indexes=[int(x.strip()) for x in raw_indexes.split(',') if x.strip()]
        upper=125 if split=='train' else 2
        if any(i<0 or i>upper for i in indexes):raise ValueError(f'Invalid {split} shard index; expected 0..{upper}')
        if len(indexes)>3:raise ValueError('Select at most three NAAB shards per workflow run')
        paths=[]
        for i in indexes:
            name=f'data/train-{i:05d}-of-00126.txt' if split=='train' else f'data/test-{i:05d}-of-00003.txt'
            paths.append(Path(hf_hub_download(repo_id='SLPL/naab',filename=name,repo_type='dataset',local_dir=str(shard_dir))))
        def rows():
            for path in paths:
                with path.open(encoding='utf-8',errors='replace') as f:
                    for line in f:
                        if line.strip():yield {'text':line}
        return rows()
    try:
        # Reserve validation first so exact duplicates cannot leak into training.
        if a.naab_val_rows:
            export_text(naab_lines('test',a.naab_test_shards),val,a.naab_val_rows,seen,'SLPL/naab:test')
        if a.naab_train_rows:
            train_count+=export_text(naab_lines('train',a.naab_train_shards),train,a.naab_train_rows,seen,'SLPL/naab')
        if a.fineweb_rows:
            ds=load_dataset(SOURCES['fineweb2_hq_fa']['id'],split='train',streaming=True)
            train_count+=export_text(ds,train,a.fineweb_rows,seen,'mohammadkrb/fineweb2-HQ-persian',append=train.exists())
    finally:
        shutil.rmtree(shard_dir,ignore_errors=True)
    if a.include_sft:
        sft=out/'sft.jsonl';sft_val=out/'sft_validation.jsonl';n=0;seen_s=set()
        with sft.open('w',encoding='utf-8') as f, sft_val.open('w',encoding='utf-8') as vf:
            def add_sft(dataset,source,license_name,limit):
                nonlocal n
                accepted=0
                for row in dataset:
                    msgs=to_messages(row)
                    if not msgs:continue
                    key=hashlib.sha256(json.dumps(msgs,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
                    if key in seen_s:continue
                    seen_s.add(key)
                    record={'messages':msgs,'source':source,'license':license_name}
                    target=vf if int(key[:8],16)%10==0 else f
                    target.write(json.dumps(record,ensure_ascii=False)+'\n')
                    n+=1;accepted+=1
                    if limit and accepted>=limit:break
                print(f'{source}: accepted {accepted:,} SFT records',flush=True)
            # FarsInstruct: broad multi-task Persian instruction data, Apache-2.0.
            add_sft(load_dataset(SOURCES['farsinstruct']['id'],split='train',streaming=True),
                    SOURCES['farsinstruct']['id'],'Apache-2.0',max(1,a.sft_rows//2))
            # Persian general knowledge: native Persian multi-turn messages, CC-BY-4.0.
            add_sft(load_dataset(SOURCES['persian_gk']['id'],split='train',streaming=True),
                    SOURCES['persian_gk']['id'],'CC-BY-4.0',min(5897,max(1,a.sft_rows//4)))
            # FarSense / CS_Farsi: context-grounded commonsense reasoning, CC-BY-SA-4.0.
            add_sft(load_dataset(SOURCES['cs_farsi']['id'],split='train',streaming=True),
                    SOURCES['cs_farsi']['id'],'CC-BY-SA-4.0',max(1,a.sft_rows//4))
        print(f'SFT total rows: {n:,}; held-out validation rows: {sum(1 for _ in sft_val.open(encoding="utf-8")):,}',flush=True)
    manifest={'sources':SOURCES,'train_rows_requested':a.naab_train_rows+a.fineweb_rows,'train_rows_written':train_count,
      'validation_rows_requested':a.naab_val_rows,'sft_enabled':bool(a.include_sft),'sft_rows_written':n if a.include_sft else 0,
      'notes':['Only public sources are downloaded. Local/private data is never accessed.','Deduplication is exact SHA-256 within each split.','Review source provenance, privacy, attribution and license terms before redistribution or commercial use.']}
    (out/'data_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Public data preparation completed.')
if __name__=='__main__':main()
