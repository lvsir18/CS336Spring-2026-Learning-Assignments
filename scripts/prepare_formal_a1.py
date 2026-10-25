"""Full TinyStories preparation; complete documents, train-only BPE, pinned source."""
import argparse, hashlib, json, os, re, subprocess, sys, time
from pathlib import Path
from cs336_basics.training import atomic_json

REVISION='f54c09fd23315a6f9c86f9dc80f725de7d8f9c64'
TRAIN_SHA256='6418d412de72888f52b5142c761ac21a582f7d1166f0bfbdb5f03ccfdec90443'
NAMES={'train':'TinyStoriesV2-GPT4-train.txt','val':'TinyStoriesV2-GPT4-valid.txt'}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''): h.update(chunk)
    return h.hexdigest()


def documents(path):
    buffer=''
    with path.open(encoding='utf-8') as f:
        for line in f:
            buffer+=line
            while '<|endoftext|>' in buffer:
                doc,buffer=buffer.split('<|endoftext|>',1)
                if doc.strip(): yield doc.strip()
    if buffer.strip(): yield buffer.strip()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--endpoint',default='https://huggingface.co'); a=p.parse_args()
    root=Path('data/tinystories'); source=root/'source'; source.mkdir(parents=True,exist_ok=True)
    log=root/'preparation-status.json'
    def status(stage,**extra):
        row={'stage':stage,'time_unix':time.time(),**extra}; atomic_json(log,row); print(json.dumps(row),flush=True)
    manifest={'dataset':'roneneldan/TinyStories','revision':REVISION,'endpoint':a.endpoint,'sources':{}}
    for split,name in NAMES.items():
        target=source/f'{split}.txt'; part=target.with_suffix('.txt.partial'); headers=source/f'{split}.headers'
        status('download',split=split)
        if not target.exists():
            # A prefetch may already have completed; do not redownload a full 2GB file.
            complete=split=='train' and part.exists() and part.stat().st_size==2227753162 and headers.exists()
            if not complete:
                subprocess.run(['curl','-fLsS','--retry','5','--retry-all-errors','--connect-timeout','20','--max-time','14400',
                                '-C','-','-D',str(headers),'-o',str(part),f'{a.endpoint}/datasets/roneneldan/TinyStories/resolve/{REVISION}/{name}'],check=True)
            raw=headers.read_text(errors='replace')
            commits=re.findall(r'(?im)^x-repo-commit:\s*([a-f0-9]{40})',raw)
            shas=re.findall(r'(?im)^x-linked-etag:\s*"?([a-f0-9]{64})',raw)
            if not commits or commits[-1]!=REVISION or not shas: raise ValueError('Missing or changed dataset revision/hash metadata')
            actual=digest(part)
            if actual!=shas[-1] or (split=='train' and actual!=TRAIN_SHA256): raise ValueError('Source SHA256 mismatch')
            os.replace(part,target)
        actual=digest(target)
        if split=='train' and actual!=TRAIN_SHA256: raise ValueError('Training source SHA256 mismatch')
        manifest['sources'][split]={'path':str(target),'bytes':target.stat().st_size,'sha256':actual,
                                    'official_url':f'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/{REVISION}/{name}'}
    atomic_json(root/'source-manifest.json',manifest)
    status('document_split_and_exact_validation_decontamination')
    val_hashes=set(); stats={}
    for split in ['val','train']:
        destination=root/f'{split}.txt'; temporary=destination.with_suffix('.txt.partial'); kept=excluded=0
        with temporary.open('w',encoding='utf-8',newline='\n') as f:
            for doc in documents(source/f'{split}.txt'):
                key=hashlib.sha256(doc.encode('utf-8')).digest()
                if split=='val': val_hashes.add(key)
                elif key in val_hashes: excluded+=1; continue
                f.write(doc+'\n<|endoftext|>\n'); kept+=1
        os.replace(temporary,destination)
        stats[split]={'documents':kept,'excluded_exact_val_matches':excluded,'sha256':digest(destination),'bytes':destination.stat().st_size}
    # BPE is learned exclusively from the complete filtered training corpus.
    status('train_bpe_full_training_corpus',vocab_size=10000)
    tokenizer=root/'tokenizer.json'; provenance=root/'tokenizer-provenance.json'
    expected={'training_sha256':stats['train']['sha256'],'vocab_size':10000}
    valid_cache=tokenizer.exists() and provenance.exists() and json.loads(provenance.read_text())==expected
    if not valid_cache:
        tokenizer_tmp=root/'tokenizer.json.partial'
        subprocess.run([sys.executable,'scripts/prepare_data.py','tokenizer','--input',str(root/'train.txt'),
                        '--output',str(tokenizer_tmp),'--vocab-size','10000'],check=True)
        os.replace(tokenizer_tmp,tokenizer); atomic_json(provenance,expected)
    for split in ['train','val']:
        status('tokenize',split=split)
        binary=root/f'{split}.bin'; metadata=Path(str(binary)+'.json')
        token_provenance=root/f'{split}-token-provenance.json'
        identity={'text_sha256':stats[split]['sha256'],'tokenizer_sha256':digest(tokenizer)}
        valid_tokens=binary.exists() and metadata.exists() and token_provenance.exists() and json.loads(token_provenance.read_text())==identity
        if not valid_tokens:
            binary_tmp=root/f'{split}.bin.partial'
            subprocess.run([sys.executable,'scripts/prepare_data.py','tokenize','--input',str(root/f'{split}.txt'),
                            '--tokenizer',str(tokenizer),'--output',str(binary_tmp)],check=True)
            os.replace(binary_tmp,binary); os.replace(Path(str(binary_tmp)+'.json'),metadata)
            atomic_json(token_provenance,identity)
        meta=json.loads(metadata.read_text()); expected=meta['tokens']*(2 if meta['dtype']=='uint16' else 4)
        if binary.stat().st_size!=expected: raise ValueError('Truncated token array')
        stats[split].update(tokens=meta['tokens'],token_sha256=digest(binary),dtype=meta['dtype'])
    manifest.update(processed=stats,tokenizer={'path':str(tokenizer),'sha256':digest(tokenizer),'vocab_size':10000},
                    segmentation='complete EOS-separated documents for split filtering; tokenization preserves independent line segmentation')
    atomic_json(root/'manifest.json',manifest); status('complete',manifest=str(root/'manifest.json'))


if __name__=='__main__': main()
