"""Single-device generation/evaluation utilities; HF backend, no vLLM dependency."""
import json,os,re
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .core import tokenize_prompt_and_output,get_response_log_probs,pack_rollout_ids


def load_rows(path):
    # JSONL records end at LF/CRLF. Unicode separators inside JSON strings
    # (e.g. GSM8K's U+2028) are content, although str.splitlines splits them.
    with Path(path).open(encoding='utf-8') as source:
        rows=[json.loads(line) for line in source if line.strip()]
    if not rows: raise ValueError('Dataset is empty')
    return rows


def prompt_for(question,template='r1_zero'):
    path=Path(__file__).parent/'prompts'/f'{template}.prompt'
    return path.read_text(encoding='utf-8').format(question=question)


def format_prompt(tokenizer,question,template='r1_zero',chat_template=False):
    text=prompt_for(question,template)
    if chat_template:
        return tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True)
    return text


def load_model(name,device='cuda',dtype='float32',checkpointing=False):
    tokenizer=AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(name,dtype=getattr(torch,dtype),attn_implementation='sdpa').to(device)
    model.config.use_cache=False
    if checkpointing: model.gradient_checkpointing_enable()
    # Keep dropout disabled in both generation and differentiable training.
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout): module.p=0.
    model.train(checkpointing)
    return model,tokenizer


@torch.no_grad()
def sample(model,tokenizer,prompts,max_new_tokens=512,batch_size=8,template='r1_zero',return_encoded=False):
    previous=tokenizer.padding_side;tokenizer.padding_side='left'
    was_training=model.training;model.eval()
    outputs=[];lengths=[];prompt_ids=[];response_ids=[]
    try:
        for start in range(0,len(prompts),batch_size):
            batch=tokenizer(prompts[start:start+batch_size],padding=True,add_special_tokens=False,return_tensors='pt').to(next(model.parameters()).device)
            generated=model.generate(**batch,do_sample=True,temperature=1.,top_p=1.,max_new_tokens=max_new_tokens,
                                     use_cache=True,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
            for index,ids in enumerate(generated[:,batch.input_ids.shape[1]:]):
                # Count emitted IDs up to EOS, instead of decode/re-encode length.
                raw_ids=ids.tolist();eos=tokenizer.eos_token_id
                emitted=raw_ids.index(eos)+1 if eos is not None and eos in raw_ids else len(raw_ids)
                if return_encoded:
                    prompt_ids.append(batch.input_ids[index][batch.attention_mask[index].bool()].tolist())
                    response_ids.append(raw_ids[:emitted])
                text=tokenizer.decode(ids,skip_special_tokens=True)
                if template!='question_only' and '</answer>' in text:
                    text=text[:text.index('</answer>')+len('</answer>')]
                outputs.append(text);lengths.append(emitted)
            if os.getenv('CS336_PROGRESS')=='1':print('GENERATION_BATCH',min(start+batch_size,len(prompts)),len(prompts),flush=True)
    finally:
        tokenizer.padding_side=previous
        model.train(was_training)
    if return_encoded:return outputs,lengths,pack_rollout_ids(prompt_ids,response_ids,tokenizer.pad_token_id)
    return outputs,lengths


@torch.no_grad()
def old_scores(model,tokenizer,prompts,responses,microbatch=1,encoded=None):
    data=encoded if encoded is not None else tokenize_prompt_and_output(prompts,responses,tokenizer)
    device=next(model.parameters()).device
    return torch.cat([get_response_log_probs(model,data['input_ids'][i:i+microbatch].to(device),data['labels'][i:i+microbatch].to(device))['log_probs'].cpu()
                      for i in range(0,len(prompts),microbatch)])


def reward_function(template):
    from .drgrpo_grader import r1_zero_reward_fn,question_only_reward_fn
    return question_only_reward_fn if template=='question_only' else r1_zero_reward_fn


@torch.no_grad()
def evaluate(model,tokenizer,rows,output,template='r1_zero',max_new_tokens=512,batch_size=8,chat_template=False):
    prompts=[format_prompt(tokenizer,r['question'],template,chat_template) for r in rows]
    responses,lengths=sample(model,tokenizer,prompts,max_new_tokens,batch_size,template)
    reward_fn=reward_function(template)
    scores=[reward_fn(y,r['answer']) for y,r in zip(responses,rows)]
    # A diagnostic independent of the exact think/answer whitespace contract.
    # Accept explicit answer tags or boxed answers, never an arbitrary last number.
    from .drgrpo_grader import grade,extract_answer
    for response,row,length,score in zip(responses,rows,lengths,scores):
        tags=re.findall(r'<answer>(.*?)</answer>',response,re.S)
        answer=tags[-1].strip() if tags else extract_answer(response)
        score.update(answer_parseable=float(bool(answer)),answer_only_reward=float(bool(answer) and grade(answer,str(row['answer']),True)),
                     hit_token_limit=float(length>=max_new_tokens))
    p=Path(output);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8') as f:
        for row,response,length,score in zip(rows,responses,lengths,scores):
            f.write(json.dumps({'question':row['question'],'ground_truth':row['answer'],'response':response,'response_tokens':length,**score},ensure_ascii=False)+'\n')
    stats={k:sum(s.get(k,0) for s in scores)/len(scores) for k in ['reward','format_reward','answer_reward','answer_parseable','answer_only_reward','hit_token_limit']}
    stats.update(examples=len(rows),mean_response_tokens=sum(lengths)/len(lengths),max_response_tokens=max(lengths))
    p.with_suffix('.summary.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    return stats
