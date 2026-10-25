import argparse
from pathlib import Path
import torch
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--tokenizer',required=True);parser.add_argument('--prompt',default='Once upon a time');parser.add_argument('--device',default='cuda');parser.add_argument('--max-tokens',type=int,default=128);parser.add_argument('--temperature',type=float,default=1.);parser.add_argument('--top-p',type=float,default=.95);parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    if args.temperature<0 or not 0<args.top_p<=1:raise ValueError('Invalid sampling settings')
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=TransformerLM(**state['config']['model']).to(args.device);model.load_state_dict(state['model']);model.eval()
    tok=Tokenizer.load(args.tokenizer);ids=torch.tensor(tok.encode(args.prompt),device=args.device)[None,:]
    if ids.numel()==0:raise ValueError('Prompt must be nonempty')
    eos=tok.inverse.get(b'<|endoftext|>');torch.manual_seed(args.seed)
    with torch.no_grad():
        for _ in range(args.max_tokens):
            logits=model(ids[:,-model.context_length:])[:,-1,:].float()
            if args.temperature==0:next_id=logits.argmax(-1,keepdim=True)
            else:
                probabilities=torch.softmax(logits/args.temperature,-1)
                values,indices=probabilities.sort(descending=True)
                remove=values.cumsum(-1)-values>=args.top_p
                values=values.masked_fill(remove,0);values/=values.sum(-1,keepdim=True)
                next_id=indices.gather(-1,torch.multinomial(values,1))
            ids=torch.cat((ids,next_id),-1)
            if next_id.item()==eos:break
    print(tok.decode(ids[0].tolist()))

if __name__=='__main__':main()
