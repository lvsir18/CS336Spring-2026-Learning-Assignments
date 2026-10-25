import numpy as np
import torch


def get_batch(dataset,batch_size,context_length,device):
    if len(dataset) <= context_length:
        raise ValueError('Dataset must have at least context_length+1 tokens')
    starts = np.random.randint(0,len(dataset)-context_length,size=batch_size)
    ids = starts[:,None]+np.arange(context_length+1)[None,:]
    batch = torch.as_tensor(np.asarray(dataset[ids],dtype=np.int64),device=device)
    return batch[:,:-1],batch[:,1:]


def save_checkpoint(model,optimizer,iteration,out):
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'iteration':iteration},out)


def load_checkpoint(src,model,optimizer):
    state = torch.load(src,map_location=next(model.parameters()).device,weights_only=False)
    model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
    return state['iteration']
