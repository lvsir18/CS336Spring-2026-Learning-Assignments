"""Versioned A4 controls and resumable multi-shard expansion; no old data edits."""
import argparse,gzip,hashlib,json,math,os,random,shutil,subprocess
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
from tokenizers import Tokenizer as FastTokenizer,AddedToken
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from cs336_basics.training import atomic_json,run_lock
from cs336_basics.tokenizer import Tokenizer
from cs336_data.filters import identify_language,gopher_quality_filter,classify_nsfw,classify_toxic_speech,mask_emails,mask_ips,mask_phone_numbers
from process_crawl import records
from fetch_formal_assets import fetch,sha256

ROOT=Path('data/optimization-v1')


def read(path):
    with Path(path).open(encoding='utf-8') as f:return [json.loads(line) for line in f if line.strip()]


def mask(text):
    for function in [mask_emails,mask_phone_numbers,mask_ips]:text,_=function(text)
    return text


def tokenizer():
    vocab='assets/gpt2/gpt2_vocab.json';merges='assets/gpt2/gpt2_merges.txt'
    fast=FastTokenizer(BPE.from_file(vocab,merges));fast.pre_tokenizer=ByteLevel(add_prefix_space=False)
    fast.add_special_tokens([AddedToken('<|endoftext|>',special=True)])
    reference=Tokenizer.from_files(vocab,merges,['<|endoftext|>'])
    # A faster backend is accepted only after checking real texts against our BPE.
    for row in read('data/a4-formal/raw.jsonl')[:20]:
        text=mask(row['text']);assert fast.encode(text).ids==reference.encode(text),'GPT2 tokenizer mismatch'
    return fast


def write_binary(path,rows,budget=None):
    temporary=Path(str(path)+'.partial');count=0;ids=[];partial=False
    with temporary.open('wb') as f:
        for row in rows:
            data=row['tokens'];remaining=budget-count if budget is not None else len(data)
            if remaining<=0:break
            partial=len(data)>remaining;data=data[:remaining]
            np.asarray(data,dtype='uint16').tofile(f);count+=len(data);ids.append(row['id'])
        f.flush();os.fsync(f.fileno())
    if budget is not None and count!=budget:raise ValueError('Not enough tokens for matched subset')
    os.replace(temporary,path);atomic_json(Path(str(path)+'.json'),{'dtype':'uint16','tokens':count,'vocab_size':50257})
    info={'tokens':count,'documents_touched':len(ids),'partial_final_document':partial,'sha256':sha256(path)}
    atomic_json(path.with_suffix('.selection.json'),info|{'source_ids':ids})
    return info


def small():
    out=ROOT/'a4-small';out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():return
    fast=tokenizer();original=read('data/a4-formal/raw.jsonl')
    for row in original:row['tokens']=fast.encode(mask(row['text'])).ids+[50256]
    scores={r['id']:r['score'] for r in read('data/a4-formal/quality-scores.jsonl')}
    clean_ids={r['id'] for r in read('data/a4-formal/clean.jsonl')}
    rules=[r for r in original if r['id'] in scores];ranked=[r for r in original if r['id'] in clean_ids]
    target=sum(len(r['tokens']) for r in ranked);variants={'raw':original,'rules':rules,'ranked':ranked}
    for seed in [42,43,44]:
        selected=list(rules);random.Random(seed).shuffle(selected);variants['random'+str(seed)]=selected
    stats={}
    for name,rows in variants.items():stats[name]=write_binary(out/(name+'.bin'),rows,target if name.startswith('random') else None)
    validation=read('data/a4-formal/val.jsonl')
    for row in validation:row['tokens']=fast.encode(mask(row['text'])).ids+[50256]
    stats['val']=write_binary(out/'val.bin',validation)
    atomic_json(out/'manifest.json',{'variants':stats,'matched_subset_tokens':target,'pii':'same masking in every training arm and common validation',
        'quality':'original disjoint Wiki proxy; ranked exactly the original selected document IDs',
        'tokenizer':'GPT2 native BPE checked on 20 documents against our implementation'})
    print('A4_SMALL_READY',json.dumps(stats),flush=True)


def expanded(target,max_shards):
    out=ROOT/'a4-expanded';out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():return
    fast=tokenizer();os.environ.update(CS336_LID_MODEL='models/formal/lid.176.bin',CS336_NSFW_MODEL='models/formal/nsfw/model.bin',CS336_TOXIC_MODEL='models/formal/toxic/model.bin')
    paths=gzip.decompress(Path('data/formal-assets/wet.paths.gz').read_bytes()).decode().splitlines()[1:]
    random.Random(42).shuffle(paths);paths=paths[:max_shards]
    heldout={urlparse(r['url']).netloc.lower().removeprefix('www.') for r in read('data/a4-formal/val.jsonl')}
    excluded={r['id'] for r in read('data/a4-formal/quality-negative.jsonl')}
    cache=out/'shards';cache.mkdir(exist_ok=True);downloads=out/'downloads';downloads.mkdir(exist_ok=True)
    seen=set();counts={'raw':0,'rules':0};documents={'raw':0,'rules':0};sources=[]
    handles={name:Path(str(out/(name+'.bin'))+'.partial').open('wb') for name in counts}
    try:
        for index,source in enumerate(paths):
            processed=cache/f'{index:03d}.jsonl';meta=processed.with_suffix('.meta.json')
            if not processed.exists() or not meta.exists():
                compressed=downloads/f'{index:03d}.wet.gz'
                download=fetch('https://data.commoncrawl.org/'+source,compressed)
                temporary=processed.with_suffix('.partial');scanned=0;kept=0
                with temporary.open('w',encoding='utf-8') as dest:
                    for url,text in records(compressed,'wet'):
                        scanned+=1;host=urlparse(url).netloc.lower().removeprefix('www.')
                        if host in heldout or int(hashlib.sha256(host.encode()).hexdigest()[:8],16)%10==0:continue
                        if hashlib.sha256(text.encode()).hexdigest() in excluded:continue
                        text=text[:200000];lang,prob=identify_language(text)
                        if lang!='en' or prob<.7:continue
                        good=gopher_quality_filter(text)
                        if good:
                            label,confidence=classify_nsfw(text);good=not(label=='nsfw' and confidence>=.9)
                        if good:
                            label,confidence=classify_toxic_speech(text);good=not(label in ('toxic','hatespeech') and confidence>=.9)
                        tokens=fast.encode(mask(text)).ids[:8192]+[50256]
                        digest=hashlib.sha256(np.asarray(tokens,dtype='uint16').tobytes()).hexdigest()
                        dest.write(json.dumps({'id':digest,'tokens':tokens,'rules':good})+'\n');kept+=1
                    dest.flush();os.fsync(dest.fileno())
                os.replace(temporary,processed);atomic_json(meta,{'source':source,'download':download,'scanned':scanned,'english_train':kept,'cache_sha256':sha256(processed)})
            information=json.loads(meta.read_text());assert information['source']==source
            assert sha256(processed)==information['cache_sha256'];sources.append(information)
            with processed.open() as src:
                for line in src:
                    row=json.loads(line)
                    if row['id'] in seen:continue
                    seen.add(row['id']);data=np.asarray(row['tokens'],dtype='uint16')
                    for name,accept in [('raw',True),('rules',row['rules'])]:
                        if accept:data.tofile(handles[name]);counts[name]+=len(data);documents[name]+=1
            atomic_json(out/'progress.json',{'shards':index+1,'tokens':counts,'documents':documents,'target_rules_tokens':target})
            print('EXPANSION',index+1,json.dumps(counts),flush=True)
            if counts['rules']>=target:break
        if counts['rules']<target:raise RuntimeError(f'Only {counts["rules"]} clean tokens after {max_shards} shards; expansion incomplete, downstream jobs will not run')
        for f in handles.values():f.flush();os.fsync(f.fileno())
    finally:
        for f in handles.values():f.close()
    for name in counts:
        path=out/(name+'.bin');os.replace(Path(str(path)+'.partial'),path)
        atomic_json(Path(str(path)+'.json'),{'dtype':'uint16','tokens':counts[name],'vocab_size':50257})
    for suffix in ['', '.json']:shutil.copyfile(str(ROOT/'a4-small/val.bin')+suffix,str(out/'val.bin')+suffix)
    atomic_json(out/'manifest.json',{'tokens':counts,'documents':documents,'shards':sources,'sha256':{name:sha256(out/(name+'.bin')) for name in ['raw','rules','val']},
        'selection':'language/exact token-document dedup; rules arm adds readability and NSFW/toxic filters, no Wiki ranking',
        'scope':'WET multi-shard; max 200000 characters/8192 GPT2 tokens per document in both arms; uniform PII masking',
        'validation':'same frozen held-out host set as small ablations; additional normalized-host hash exclusion; not registrable-domain isolation'})


def main():
    p=argparse.ArgumentParser();p.add_argument('--part',choices=['small','expanded'],required=True);p.add_argument('--target-tokens',type=int,default=100000000);p.add_argument('--max-shards',type=int,default=64);a=p.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    with run_lock(ROOT):
        if a.part=='small':small()
        else:expanded(a.target_tokens,a.max_shards)


if __name__=='__main__':main()
