"""Real Common Crawl ablation data and held-out GSM8K; no synthetic task substitutions."""
import argparse,collections,decimal,hashlib,json,math,os,random,subprocess,sys
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from cs336_basics.training import atomic_json,run_lock
from cs336_basics.tokenizer import Tokenizer
from cs336_alignment.hf_workflow import prompt_for
from cs336_data.filters import identify_language,gopher_quality_filter,classify_nsfw,classify_toxic_speech,classify_quality,mask_emails,mask_ips,mask_phone_numbers
from cs336_data.dedup import minhash_deduplication
from process_crawl import records


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_rows(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.partial')
    with temp.open('w',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    os.replace(temp,path)


def gsm8k():
    root=Path('data/gsm8k-formal');root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists():return
    tok=AutoTokenizer.from_pretrained('models/olmo-2-0425-1b')
    train=pq.read_table('data/formal-assets/gsm8k/train-00000-of-00001.parquet').to_pylist()
    test=pq.read_table('data/formal-assets/gsm8k/test-00000-of-00001.parquet').to_pylist()
    ids=list(range(len(train)));random.Random(42).shuffle(ids);val_ids=set(ids[:256])
    splits={'train':[],'val':[],'test':[]};sft=[];dpo=[];rejected=0
    for name,source in [('train',train),('test',test)]:
        for i,row in enumerate(source):
            rationale,final=row['answer'].rsplit('####',1);final=final.strip()
            target='val' if name=='train' and i in val_ids else name
            result={'id':f'{name}-{i}','question':row['question'],'answer':final,'solution':row['answer']}
            splits[target].append(result)
            if target!='train':continue
            response=rationale.strip()+'</think> <answer>'+final+'</answer>'
            length=len(tok.encode(prompt_for(row['question']),add_special_tokens=False))+len(tok.encode(response+tok.eos_token,add_special_tokens=False))-1
            if length>1024:rejected+=1;continue
            sft.append({**result,'response':response})
            try:wrong=str(decimal.Decimal(final.replace(',',''))+1)
            except decimal.InvalidOperation:continue
            negative=rationale.strip()+'</think> <answer>'+wrong+'</answer>'
            if len(tok.encode(prompt_for(row['question'])+negative+tok.eos_token,add_special_tokens=False))>1025:continue
            dpo.append({'question':row['question'],'chosen':response,'rejected':negative,'source_id':result['id'],
                        'preference_origin':'constructed incorrect final number; not human safety preference'})
    for name,rows in splits.items():write_rows(root/f'{name}.jsonl',rows)
    write_rows(root/'sft.jsonl',sft);write_rows(root/'dpo.jsonl',dpo)
    atomic_json(root/'manifest.json',{'source':'openai/gsm8k public main parquet','seed':42,
        'counts':{**{k:len(v) for k,v in splits.items()},'sft':len(sft),'dpo':len(dpo)},'overlength_sft_excluded':rejected,
        'sha256':{p.name:digest(p) for p in root.glob('*.jsonl')},'dpo':'synthetically constructed math consistency preferences, separate from RL branch'})
    print('GSM8K_READY',flush=True)


def web():
    root=Path('data/a4-formal');root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists():return
    os.environ.update(CS336_LID_MODEL='models/formal/lid.176.bin',CS336_NSFW_MODEL='models/formal/nsfw/model.bin',
                      CS336_TOXIC_MODEL='models/formal/toxic/model.bin',CS336_QUALITY_MODEL='models/formal/quality.bin')
    # Independently exercise HTML extraction; LM ablation below uses WET text.
    html=[]
    for url,text in records('data/formal-assets/example.warc.gz','warc'):
        html.append({'url':url,'text':text})
        if len(html)>=200:break
    write_rows(root/'html-extraction-sample.jsonl',html)
    pools={'train':[],'val':[]};seen=set();stats=collections.Counter()
    for url,text in records('data/formal-assets/example.wet.gz','wet'):
        stats['input_documents']+=1
        if stats['input_documents']>150000:break
        key=hashlib.sha256(text.encode()).hexdigest()
        if key in seen:stats['exact_duplicate']+=1;continue
        seen.add(key)
        lang,prob=identify_language(text)
        if lang!='en' or prob<.7:stats['non_english']+=1;continue
        domain=urlparse(url).netloc or key
        split='val' if int(hashlib.sha256(domain.encode()).hexdigest()[:8],16)%10==0 else 'train'
        pools[split].append({'url':url,'text':text,'id':key,'domain':domain})
        if sum(map(len,pools.values()))>=30000:break
    if len(pools['train'])<1000 or len(pools['val'])<100:raise ValueError('Too little real web data in selected shard')
    print('WEB_SPLIT',json.dumps({k:len(v) for k,v in pools.items()}),flush=True)
    wiki=[]
    for row in pq.read_table('data/formal-assets/wiki/train-00000-of-00001.parquet').to_pylist():
        if len(row['text'].split())>=50:wiki.append({'text':row['text'],'source':'WikiText-2 raw train proxy'})
        if len(wiki)>=3000:break
    # Reserve independent classifier-reference documents from both LM arms.
    reference_count=min(1000,len(pools['train'])//5,len(wiki))
    quality_negative=pools['train'][:reference_count];pools['train']=pools['train'][reference_count:]
    write_rows(root/'wiki-positive.jsonl',wiki[:reference_count]);write_rows(root/'quality-negative.jsonl',quality_negative)
    subprocess.run([sys.executable,'scripts/train_quality_classifier.py','--positive',str(root/'wiki-positive.jsonl'),
                    '--negative',str(root/'quality-negative.jsonl'),'--output','models/formal/quality.bin'],check=True)
    import fasttext
    for name,path in [('nsfw','models/formal/nsfw/model.bin'),('toxic','models/formal/toxic/model.bin')]:
        labels=fasttext.load_model(path).get_labels();print('CLASSIFIER_LABELS',name,labels,flush=True)
        normalized={label.removeprefix('__label__') for label in labels}
        if not normalized.intersection({'nsfw'} if name=='nsfw' else {'toxic','hatespeech'}):raise ValueError(f'Unexpected {name} classifier labels')
    quality_model=fasttext.load_model('models/formal/quality.bin');candidates=[]
    for row in pools['train']:
        text=row['text']
        if not gopher_quality_filter(text):stats['reject_rules']+=1;continue
        label,confidence=classify_nsfw(text)
        if label=='nsfw' and confidence>=.9:stats['reject_nsfw']+=1;continue
        label,confidence=classify_toxic_speech(text)
        if label in ('toxic','hatespeech') and confidence>=.9:stats['reject_toxic']+=1;continue
        labels,probabilities=quality_model.predict(' '.join(text.split()),k=-1)
        confidence=float(probabilities[list(labels).index('__label__wiki')])
        for function in [mask_emails,mask_phone_numbers,mask_ips]:text,_=function(text)
        candidates.append({**row,'text':text,'quality_wiki_score':confidence})
    # Source-proxy confidence is not calibrated web quality. Use a prespecified
    # rank fraction on unseen LM documents, without consulting validation loss.
    candidates.sort(key=lambda row:(-row['quality_wiki_score'],row['id']))
    clean=candidates[:math.ceil(.3*len(candidates))];stats['reject_quality_rank']=len(candidates)-len(clean)
    write_rows(root/'quality-scores.jsonl',[{'id':row['id'],'score':row['quality_wiki_score']} for row in candidates])
    print('FILTERED_DOCUMENTS',len(clean),json.dumps(stats),flush=True)
    if len(clean)<200:raise ValueError('Quality filter retained fewer than 200 documents; inspect thresholds before training')
    fingerprint=hashlib.sha256('\n'.join(row['id'] for row in clean).encode()).hexdigest()[:16]
    source=root/'dedup-input';destination=root/f'dedup-output-{fingerprint}';source.mkdir(exist_ok=True)
    for i,row in enumerate(clean):(source/f'{i:08d}.txt').write_text(row['text'],encoding='utf-8')
    dedup=minhash_deduplication([source/f'{i:08d}.txt' for i in range(len(clean))],100,20,5,.8,destination)
    retained=[row for i,row in enumerate(clean) if (destination/f'{i:08d}.txt').exists()]
    validation=[row for row in pools['val'] if gopher_quality_filter(row['text'])]
    if len(validation)<50:raise ValueError('Too few held-out domains for web validation')
    variants={'raw':pools['train'],'clean':retained,'val':validation}
    tok=Tokenizer.from_files('assets/gpt2/gpt2_vocab.json','assets/gpt2/gpt2_merges.txt',['<|endoftext|>'])
    counts={}
    for name,rows in variants.items():
        write_rows(root/f'{name}.jsonl',rows);binary=root/f'{name}.bin';temporary=binary.with_suffix('.bin.partial');count=0
        with temporary.open('wb') as f:
            for row in rows:
                ids=tok.encode(row['text'])+[50256];np.asarray(ids,dtype='uint16').tofile(f);count+=len(ids)
        os.replace(temporary,binary)
        atomic_json(Path(str(binary)+'.json'),{'dtype':'uint16','tokens':count,'vocab_size':50257,'segmentation':'whole document plus GPT2 EOS'})
        counts[name]={'documents':len(rows),'tokens':count,'sha256':digest(binary)}
    atomic_json(root/'manifest.json',{'variants':counts,'filters':dict(stats),'dedup':dedup,'domain_split':'sha256(domain)%10==0 validation',
        'quality_reference':'WikiText-2 train proxy plus disjoint reserved CC train negatives; excluded from both LM arms',
        'quality_policy':{'rule_survivors':len(candidates),'retained_fraction':.3,'classifier_reference_per_class':reference_count,
                          'selection':'top Wiki-proxy score; not calibrated probability of true quality'},
        'source':'CC-MAIN-2026-12 first WET shard; separate 200-page WARC extraction sample','gpt2_vocab_sha256':digest('assets/gpt2/gpt2_vocab.json'),
        'quality_model_sha256':digest('models/formal/quality.bin')})
    print('WEB_DATA_READY',json.dumps(counts),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--part',choices=['gsm8k','web','all'],default='all');args=p.parse_args()
    root=Path('data/remaining-preparation');root.mkdir(parents=True,exist_ok=True)
    with run_lock(root):
        if args.part in ['gsm8k','all']:gsm8k()
        if args.part in ['web','all']:web()
