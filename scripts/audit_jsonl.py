"""Memory-bounded JSONL audit: counts, Persian ratio, exact duplicate estimate."""
import argparse,json,hashlib
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--limit',type=int,default=0);a=p.parse_args()
 path=Path(a.input);seen=set();rows=bad=dupes=chars=fa=letters=0;short=0
 with path.open(encoding='utf-8',errors='replace') as f:
  for line in f:
   rows+=1
   try:r=json.loads(line)
   except ValueError:bad+=1;continue
   text=r.get('text','') if isinstance(r,dict) else ''
   if not isinstance(text,str):bad+=1;continue
   text=text.strip();chars+=len(text)
   ls=[c for c in text if c.isalpha()];letters+=len(ls);fa+=sum('\u0600'<=c<='\u06ff' for c in ls)
   if len(text)<40:short+=1
   h=hashlib.sha256(text.encode()).digest()
   if h in seen:dupes+=1
   else:seen.add(h)
   if a.limit and rows>=a.limit:break
 report={'file':str(path),'rows_scanned':rows,'invalid_or_nontext':bad,'short_texts_under_40_chars':short,'exact_duplicates':dupes,'characters':chars,'persian_letter_ratio':round(fa/max(1,letters),4)}
 print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
