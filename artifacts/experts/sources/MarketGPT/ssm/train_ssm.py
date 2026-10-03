"""
This script is based on nanoGPT by Karpathy: https://github.com/karpathy/nanoGPT

This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
from os.path import dirname, abspath
import sys

ROOT_DIR = dirname(dirname(abspath(__file__)))
sys.path.append(ROOT_DIR)

import time
import math
import random
import inspect
from glob import glob
# import pickle
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from equities.data_processing.itch_encoding import Vocab, encode_msgs
# # from model import GPTConfig, GPT
from lm import LM
# # from ssm.transformer.transformer import TransformerConfig
from mamba import MambaConfig
# from jamba import JambaConfig

# # official mamba package (20% faster per iteration and 40% less memory usage w/ 48 layers and 1024 embed dim)
# from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
# from mamba_ssm.models.config_mamba import MambaConfig

# -----------------------------------------------------------------------------
# default config values
# I/O
out_dir = 'out'
checkpoint_name = 'ckpt_ssm_v3.pt'
eval_interval = 50 # 2000
log_interval = 1
eval_iters = 100 # 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume'
# wandb logging
wandb_log = True # False # disabled by default
wandb_project = 'MarketSimSSM'
wandb_run_name = 'run' + str(time.time())
# data
seed = 42
rng = random.Random(seed)
torch.manual_seed(seed)
msg_seq_len = 112 # 112 # 432
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 1 # 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 2687 # 2687 # 10367
# model
n_layer = 48 # 24
# n_head = 12
n_embd = 1536 # 768
# dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
# bias = False # do we use bias inside LayerNorm and Linear layers?
use_cuda_mamba = True # use CUDA implementation of Mamba
# adamw optimizer
learning_rate = 1e-3 # max learning rate
max_iters = 4000 # 600000 # total number of training iterations
weight_decay = 0.01 # 0.00001
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 50 # 2000 # how many steps to warm up for
lr_decay_iters = max_iters # 1000 # 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# exec(open('equities/configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
train_data_dir = os.path.join(os.path.abspath(''), 'dataset/proc/ITCH/train/')
train_message_files = sorted(glob(str(train_data_dir) + '/*message*.npy'))
assert len(train_message_files) > 0, f'no message files found in {train_data_dir}'
val_data_dir = os.path.join(os.path.abspath(''), 'dataset/proc/ITCH/val/')
val_message_files = sorted(glob(str(val_data_dir) + '/*message*.npy'))
assert len(val_message_files) > 0, f'no message files found in {val_data_dir}'
test_data_dir = os.path.join(os.path.abspath(''), 'dataset/proc/ITCH/test/')
test_message_files = sorted(glob(str(test_data_dir) + '/*message*.npy'))
assert len(test_message_files) > 0, f'no message files found in {test_data_dir}'
train_datasets = []
for file in train_message_files:
    train_datasets.append(np.load(file, mmap_mode='r'))
val_datasets = []
for file in val_message_files:
    val_datasets.append(np.load(file, mmap_mode='r'))
test_datasets = []
for file in test_message_files:
    test_datasets.append(np.load(file, mmap_mode='r'))
vocab = Vocab()
def get_batch(split):
    datasets = train_datasets if split == 'train' else val_datasets
    data = rng.choice(datasets)
    ix = torch.randint(len(data) - msg_seq_len, (batch_size,))
    x = torch.stack([torch.from_numpy((encode_msgs((data[i:i+msg_seq_len]).astype(np.int64), vocab.ENCODING)).reshape(-1)) for i in ix])
    # target y is the same as x but shifted by one token
    y = x[:, 1:]
    y = y.type(torch.LongTensor) # casting to long for cross entropy loss fn
    x = x[:, :-1]
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

meta_vocab_size = len(vocab)
print(f"using vocab_size = {meta_vocab_size}")

# model init
# model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
#                   bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
model_args = dict(d_model=n_embd, n_layers=n_layer, use_cuda=use_cuda_mamba)
# model_args = dict(d_model=n_embd, n_layer=n_layer, vocab_size=meta_vocab_size) # official mamba package
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of 12544 (12515 rounded up for efficiency)")
    # model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 12515
    # gptconf = GPTConfig(**model_args)
    # model = GPT(gptconf)
    model_config = MambaConfig(**model_args)
    model = LM(model_config, vocab_size=meta_vocab_size)

    # # official mamba package
    # model_config = MambaConfig(**model_args)
    # model = MambaLMHeadModel(model_config)
# elif init_from == 'resume':
#     print(f"Resuming training from {out_dir}")
#     # resume training from a checkpoint.
#     ckpt_path = os.path.join(out_dir, 'ckpt.pt')
#     checkpoint = torch.load(ckpt_path, map_location=device)
#     checkpoint_model_args = checkpoint['model_args']
#     # force these config attributes to be equal otherwise we can't even resume training
#     # the rest of the attributes (e.g. dropout) can stay as desired from command line
#     for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
#         model_args[k] = checkpoint_model_args[k]
#     # create the model
#     gptconf = GPTConfig(**model_args)
#     model = GPT(gptconf)
#     state_dict = checkpoint['model']
#     # fix the keys of the state dictionary :(
#     # honestly no idea how checkpoints sometimes get this prefix, have to debug more
#     unwanted_prefix = '_orig_mod.'
#     for k,v in list(state_dict.items()):
#         if k.startswith(unwanted_prefix):
#             state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
#     model.load_state_dict(state_dict)
#     iter_num = checkpoint['iter_num']
#     best_val_loss = checkpoint['best_val_loss']
# # crop down the model block size if desired, using model surgery
# if block_size < model.config.block_size:
#     model.crop_block_size(block_size)
#     model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)
print("Model initialized. Number of parameters: %.2fM" % (sum([p.numel() for p in model.parameters()])/1e6,))
# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# # optimizer
# def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
#     param_dict = {pn: p for pn, p in model.named_parameters()}
#     param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
#     # any parameters that is 2D will be weight decayed, otherwise no. (i.e. all weight tensors in matmuls + embeddings decay, all biases and rmnsnorms don't)
#     decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
#     nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
#     optim_groups = [
#         {'params': decay_params, 'weight_decay': weight_decay},
#         {'params': nodecay_params, 'weight_decay': 0.0}
#     ]
#     # Create AdamW optimizer and use the fused version if it is available
#     fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
#     use_fused = fused_available and device_type == 'cuda'
#     extra_args = dict(fused=True) if use_fused else dict()
#     optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
#     return optimizer

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
# optimizer = configure_optimizers(model, weight_decay, learning_rate, (beta1, beta2), device_type)
# if init_from == 'resume':
#     optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits = model(X)
                # logits = model(X).logits # official mamba package
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=0)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model # the unoptimized model is kept for saving
# running_mfu = -1.0
print("Training is starting.")
start_time = time.time()
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "tokens": iter_num * tokens_per_iter,
                "loss/train": losses['train'],
                "loss/val": losses['val'],
                "lr": lr,
                # "mfu": running_mfu*100, # convert to percentage
            }, step = iter_num)
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, checkpoint_name))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits = model(X)
            # logits = model(X).logits # official mamba package
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=0)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        # if local_iter_num >= 5: # let the training loop settle a bit
        #     mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
        #     running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        # print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

end_time = time.time()
print(f"Training is done. Took {(end_time-start_time)/60:.2f} minutes.")