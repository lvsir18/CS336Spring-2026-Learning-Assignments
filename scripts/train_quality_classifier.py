"""Train on independent positive/negative JSONL documents, never on test fixtures."""
import argparse
import json
from pathlib import Path
import random

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--positive',required=True,help='Wikipedia/reference-quality JSONL, text key')
    parser.add_argument('--negative',required=True,help='Raw web JSONL, text key')
    parser.add_argument('--output',default='models/quality.bin')
    args=parser.parse_args()
    import fasttext
    rows=[]
    for source,label in [(args.positive,'wiki'),(args.negative,'cc')]:
        with open(source,encoding='utf-8') as f:
            for line in f:
                text=' '.join(json.loads(line)['text'].split())
                if text: rows.append(f'__label__{label} {text}\n')
    rng=random.Random(42);rng.shuffle(rows)
    if len(rows)<20: raise ValueError('Provide at least 20 independent training documents')
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    split=int(len(rows)*.9)
    training=path.parent/'quality-training.txt'; validation=path.parent/'quality-validation.txt'
    training.write_text(''.join(rows[:split]),encoding='utf-8');validation.write_text(''.join(rows[split:]),encoding='utf-8')
    model=fasttext.train_supervised(input=str(training),wordNgrams=2,dim=64,epoch=10,lr=.1,thread=4)
    model.save_model(str(path))
    result=model.test(str(validation))
    print({'held_out_count':result[0],'precision':result[1],'recall':result[2]})

if __name__=='__main__':main()
