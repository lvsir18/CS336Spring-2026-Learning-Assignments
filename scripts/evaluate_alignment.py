import argparse,json,torch
from cs336_alignment.hf_workflow import load_model,load_rows,evaluate
p=argparse.ArgumentParser();p.add_argument('--model',default='allenai/OLMo-2-0425-1B');p.add_argument('--data',default='data/gsm8k/test.jsonl');p.add_argument('--prompt',choices=['question_only','r1_zero','r1_zero_three_shot_gsm8k'],default='r1_zero');p.add_argument('--output',default='runs/a5/eval.jsonl');p.add_argument('--limit',type=int,default=0);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--max-new-tokens',type=int,default=512);p.add_argument('--device',default='cuda');p.add_argument('--dtype',default='bfloat16');p.add_argument('--seed',type=int,default=42);p.add_argument('--chat-template',action='store_true');a=p.parse_args()
torch.manual_seed(a.seed);model,tok=load_model(a.model,a.device,a.dtype)
rows=load_rows(a.data);rows=rows[:a.limit] if a.limit else rows
print(json.dumps(evaluate(model,tok,rows,a.output,a.prompt,a.max_new_tokens,a.batch_size,a.chat_template),indent=2))
