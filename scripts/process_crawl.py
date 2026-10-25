"""Bounded-document streaming extraction and filtering of WARC/WET/JSONL files."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from cs336_data.filters import *

def records(path,kind):
    if kind=='jsonl':
        with open(path,encoding='utf-8') as f:
            for line in f:
                row=json.loads(line);yield row.get('url',''),row['text']
        return
    from warcio.archiveiterator import ArchiveIterator
    with open(path,'rb') as f:
        for record in ArchiveIterator(f):
            expected='response' if kind=='warc' else 'conversion'
            if record.rec_type!=expected:continue
            content=record.content_stream().read()
            if kind=='warc':
                if record.http_headers and 'html' not in (record.http_headers.get_header('Content-Type') or ''):continue
                text=extract_text_from_html_bytes(content)
            else:text=content.decode('utf-8',errors='replace')
            yield record.rec_headers.get_header('WARC-Target-URI') or '',text

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',nargs='+',required=True);parser.add_argument('--kind',choices=['warc','wet','jsonl'],required=True);parser.add_argument('--output',required=True);parser.add_argument('--language',default='en');parser.add_argument('--min-language-score',type=float,default=.7);parser.add_argument('--max-toxic-score',type=float,default=.9);parser.add_argument('--max-nsfw-score',type=float,default=.9);parser.add_argument('--quality',action='store_true');parser.add_argument('--min-quality-score',type=float,default=.5);parser.add_argument('--limit',type=int,default=0);parser.add_argument('--extract-only',action='store_true')
    args=parser.parse_args();output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);stats=Counter();seen=set()
    with output.open('w',encoding='utf-8') as dest:
        for path in args.input:
            for url,text in records(path,args.kind):
                if args.limit and stats['input']>=args.limit:break
                stats['input']+=1
                if args.extract_only:
                    dest.write(json.dumps({'url':url,'text':text,'source':str(path)},ensure_ascii=False)+'\n');stats['retained']+=1;continue
                if not text or not gopher_quality_filter(text):stats['reject_rules']+=1;continue
                language,score=identify_language(text)
                if language!=args.language or score<args.min_language_score:stats['reject_language']+=1;continue
                label,confidence=classify_nsfw(text)
                if label=='nsfw' and confidence>=args.max_nsfw_score:stats['reject_nsfw']+=1;continue
                label,confidence=classify_toxic_speech(text)
                if label=='toxic' and confidence>=args.max_toxic_score:stats['reject_toxic']+=1;continue
                if args.quality:
                    label,confidence=classify_quality(text)
                    if label!='wiki' or confidence<args.min_quality_score:stats['reject_quality']+=1;continue
                digest=hashlib.sha256(text.encode('utf-8')).hexdigest()
                if digest in seen:stats['reject_exact_document']+=1;continue
                seen.add(digest)
                pii={}
                for name,function in [('email',mask_emails),('phone',mask_phone_numbers),('ip',mask_ips)]:text,pii[name]=function(text)
                dest.write(json.dumps({'url':url,'text':text,'source':str(path),'language':language,'language_score':score,'pii_counts':pii,'sha256_before_masking':digest},ensure_ascii=False)+'\n')
                stats['retained']+=1
    output.with_suffix('.stats.json').write_text(json.dumps(dict(stats),indent=2),encoding='utf-8');print(dict(stats))

if __name__=='__main__':main()
