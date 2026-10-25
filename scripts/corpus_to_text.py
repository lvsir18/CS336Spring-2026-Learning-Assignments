import argparse,gzip,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--kind',choices=['jsonl','gzip'],default='jsonl');a=p.parse_args();out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
if Path(a.input).resolve()==out.resolve():raise ValueError('Input and output must differ')
if a.kind=='gzip':
 with gzip.open(a.input,'rb') as source,out.open('wb') as dest:shutil.copyfileobj(source,dest)
else:
 with open(a.input,encoding='utf-8') as source,out.open('w',encoding='utf-8') as dest:
  for line in source:dest.write(json.loads(line)['text'].rstrip()+'\n<|endoftext|>\n')
