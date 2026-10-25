"""Small-corpus JSONL -> line dedup -> MinHash -> provenance-preserving JSONL."""
import argparse,json
from pathlib import Path
from cs336_data.dedup import exact_line_deduplication,minhash_deduplication
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--workdir',default='data/a4-dedup');p.add_argument('--threshold',type=float,default=.8);a=p.parse_args()
work=Path(a.workdir)
if work.exists():raise ValueError('Choose a fresh workdir to avoid retaining stale output files')
raw=work/'raw';raw.mkdir(parents=True);rows=[]
with open(a.input,encoding='utf-8') as f:
 for i,line in enumerate(f):
    row=json.loads(line);rows.append(row);(raw/f'{i:09d}.txt').write_text(row['text'],encoding='utf-8')
exact_line_deduplication(sorted(raw.glob('*.txt')),work/'line')
nonempty=[p for p in sorted((work/'line').glob('*.txt')) if p.read_text(encoding='utf-8').strip()]
stats=minhash_deduplication(nonempty,100,20,5,a.threshold,work/'fuzzy')
output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
with output.open('w',encoding='utf-8') as dest:
 for p in sorted((work/'fuzzy').glob('*.txt')):
    row=rows[int(p.stem)].copy();row['text']=p.read_text(encoding='utf-8');row['dedup_version']='exact-line-all-occurrences + minhash100-bands20-ngram5-seed42';dest.write(json.dumps(row,ensure_ascii=False)+'\n')
print(stats)
