import argparse
from pathlib import Path
import json
import numpy as np
from cs336_basics.tokenizer import Tokenizer,train_bpe

def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='action',required=True)
    train=sub.add_parser('tokenizer');train.add_argument('--input',required=True);train.add_argument('--output',required=True);train.add_argument('--vocab-size',type=int,default=10000)
    tokenize=sub.add_parser('tokenize');tokenize.add_argument('--input',required=True);tokenize.add_argument('--tokenizer');tokenize.add_argument('--output',required=True);tokenize.add_argument('--gpt2',action='store_true')
    args=parser.parse_args()
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    if args.action=='tokenizer':
        vocab,merges=train_bpe(args.input,args.vocab_size,['<|endoftext|>'])
        Tokenizer(vocab,merges,['<|endoftext|>']).save(path)
        print({'vocabulary_size':len(vocab),'merges':len(merges)})
        return
    if args.gpt2:
        import tiktoken
        tok=tiktoken.get_encoding('gpt2')
        encode=lambda text:tok.encode(text,allowed_special={'<|endoftext|>'})
        size=tok.n_vocab
    else:
        if not args.tokenizer: parser.error('--tokenizer is required unless --gpt2')
        tok=Tokenizer.load(args.tokenizer);encode=tok.encode;size=max(tok.vocab)+1
    dtype=np.dtype('uint16' if size<=65536 else 'uint32')
    count=byte_count=0
    with open(args.input,encoding='utf-8') as source,path.open('wb') as dest:
        for line in source:
            ids=encode(line);np.asarray(ids,dtype=dtype).tofile(dest)
            count+=len(ids);byte_count+=len(line.encode('utf-8'))
    metadata={'dtype':dtype.name,'tokens':count,'vocab_size':size,'input_utf8_bytes':byte_count,'tokenizer':args.tokenizer if not args.gpt2 else 'gpt2',
              'segmentation':'encode each input line independently; preserve its line ending'}
    path.with_suffix(path.suffix+'.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(metadata)

if __name__=='__main__':main()
