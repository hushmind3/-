"""
Based on the equities/train.py script, this script is version that works with wandb to perform
hyperparameter sweeps.

To run on a single GPU sweep run, example:
$ python3 equities/train_sweep.py config/fast_model.yaml --count=5
"""

import argparse
import math
import os
from os.path import dirname, abspath
import sys
import time
import random
import pickle as pkl
from glob import glob
from contextlib import nullcontext
from datetime import datetime
import yaml

ROOT_DIR = dirname(dirname(abspath(__file__)))
sys.path.append(ROOT_DIR)

import numpy as np
import wandb
import torch

from data_processing.itch_encoding import Vocab, encode_msgs
from fast_model import Transformer, ModelArgs


# poor man's data loader
def get_batch(split, train_datasets, val_datasets, vocab, config, rng, device_type='cuda', use_sink=True, bpe_tokenizer=None):
    # data = train_data if split == 'train' else val_data
    datasets = train_datasets if split == 'train' else val_datasets
    data = rng.choice(datasets)
    ix = torch.randint(len(data) - config.msg_seq_len, (config.batch_size,))
    if config.use_bpe:
        assert config.batch_size == 1, "batch size must be 1 for BPE encoding (for now)" # TODO: make this batch-wise
        # basic encoding of the messages
        basic_encoded = [(encode_msgs((data[i:i+config.msg_seq_len]).astype(np.int64), vocab.ENCODING)) for i in ix]
        # bpe encode the messages and concat EOM token to the end
        bpe_encoded = []
        for batch in range(len(basic_encoded)):
            for msg in range(len(basic_encoded[batch])):
                bpe_encoded = bpe_encoded + bpe_tokenizer.bpe_encode(basic_encoded[batch][msg]) + [config.eom_token_val]
        # convert to tensor, unsqueeze to add batch dimension
        x = torch.tensor(bpe_encoded).unsqueeze(0)
    else:
        # use basic encoding only
        x = torch.stack([torch.from_numpy((encode_msgs((data[i:i+config.msg_seq_len]).astype(np.int64), vocab.ENCODING)).reshape(-1)) for i in ix])
    if use_sink:
        # append sink token to start of each batch sequence (since vocab.SINK_TOK = 1, we can just use torch.ones)
        x = torch.cat([torch.ones((config.batch_size, 1), dtype=torch.int), x], dim=1)
    # target y is the same as x but shifted by one token
    y = x[:, 1:]
    y = y.type(torch.LongTensor) # casting to long for cross entropy loss fn
    x = x[:, :-1] # offset x by one (final) token to match y
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(config.device, non_blocking=True), y.pin_memory().to(config.device, non_blocking=True)
    else:
        x, y = x.to(config.device), y.to(config.device)
    return x, y

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(model, raw_model, ctx, train_datasets, val_datasets, vocab, config, rng, device_type='cuda', bpe_tokenizer=None):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(config.eval_iters)  # keep on CPU
        for k in range(config.eval_iters):
            X, Y = get_batch(split, train_datasets, val_datasets, vocab, config, rng, device_type=device_type, bpe_tokenizer=bpe_tokenizer)
            with ctx:
                logits = model(X, Y)
                loss = raw_model.last_loss
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it, config, lr_decay_iters, min_lr):
    # 1) linear warmup for warmup_iters steps
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - config.warmup_iters) / (lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (config.learning_rate - min_lr)

def train_model(config=None):

    with wandb.init(config=config) as run:
        # extract the config object associated with the run
        config = run.config

        # various inits
        out_dir = "out"
        rng = random.Random(config.seed)
        vocab = Vocab()
        vocab_size = config.vocab_size # len(vocab) # 12515
        lr_decay_iters = config.max_iters  # should be ~= max_iters per Chinchilla
        min_lr = 6e-5 / 10 # 0.0 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
        tokens_per_iter = config.gradient_accumulation_steps * config.batch_size * config.max_seq_len
        print(f"tokens per iteration will be: {tokens_per_iter:,}")
        print(f"breaks down as: {config.gradient_accumulation_steps} grad accum steps * {config.batch_size} batch size * {config.max_seq_len} max seq len")
        os.makedirs(out_dir, exist_ok=True)
        torch.manual_seed(config.seed)
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
        device_type = "cuda" if "cuda" in config.device else "cpu"  # for later use in torch.autocast
        # note: float16 data type will automatically use a GradScaler
        ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[config.dtype]
        ctx = (
            nullcontext()
            if device_type == "cpu"
            else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        )

        # locate directories with training and validation data
        train_data_dir = os.path.join(os.path.abspath(''), 'dataset/proc/ITCH/train/')
        train_message_files = sorted(glob(str(train_data_dir) + '/*message*.npy'))
        assert len(train_message_files) > 0, f'no message files found in {train_data_dir}'
        val_data_dir = os.path.join(os.path.abspath(''), 'dataset/proc/ITCH/val/')
        val_message_files = sorted(glob(str(val_data_dir) + '/*message*.npy'))
        assert len(val_message_files) > 0, f'no message files found in {val_data_dir}'
        # fill list with datasets, seperated by file (each file is a day of data)
        train_datasets = []
        for file in train_message_files:
            train_datasets.append(np.load(file, mmap_mode='r'))
        val_datasets = []
        for file in val_message_files:
            val_datasets.append(np.load(file, mmap_mode='r'))
        # BPE encoding
        if config.use_bpe:
            # load the tokenizer
            with open('tokenizer/bpe_tokenizer.pkl', 'rb') as f:
                bpe_tokenizer = pkl.load(f)
        # verify sink token compatibility with dataloader implementation
        if config.use_sink:
            assert vocab.SINK_TOK == 1

        # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
        iter_num = 0
        best_val_loss = 1e9

        # model init
        model_args = dict(
            dim=config.dim,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            vocab_size=vocab_size,
            multiple_of=config.multiple_of,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )  # start with model_args from command line

        # init a new model from scratch
        print("Initializing a new model from scratch")
        gptconf = ModelArgs(**model_args)
        model = Transformer(gptconf)
        model.to(config.device)

        # initialize a GradScaler. If enabled=False scaler is a no-op
        scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype == "float16"))

        # optimizer
        optimizer = model.configure_optimizers(config.weight_decay, config.learning_rate, (config.beta1, config.beta2), device_type)
        checkpoint = None  # free up memory

        # compile the model
        if config.compile:
            print("compiling the model... (takes a ~minute)")
            model = torch.compile(model)  # requires PyTorch 2.0

        # training loop
        X, Y = get_batch('train', train_datasets, val_datasets, vocab, config, rng, device_type=device_type, bpe_tokenizer=bpe_tokenizer) # fetch the very first batch
        t0 = time.time()
        local_iter_num = 0  # number of iterations in the lifetime of this process
        raw_model = model # keep a reference to the raw model
        running_mfu = -1.0
        while True:
            # determine and set the learning rate for this iteration
            lr = get_lr(iter_num, config, lr_decay_iters, min_lr) if config.decay_lr else config.learning_rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # evaluate the loss on train/val sets and write checkpoints
            if iter_num % config.eval_interval == 0:
                losses = estimate_loss(model, raw_model, ctx, train_datasets, val_datasets, vocab, config, rng, device_type=device_type, bpe_tokenizer=bpe_tokenizer)
                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                try:
                    wandb.log(
                        {
                            "iter": iter_num,
                            "tokens": iter_num * tokens_per_iter,
                            "loss/train": losses["train"],
                            "loss/val": losses["val"],
                            "lr": lr,
                            "mfu": running_mfu * 100,  # convert to percentage
                        }, step = iter_num
                    )
                except Exception as e:
                    print(f"logging to wandb failed: {e}")
                # if losses["val"] < best_val_loss or always_save_checkpoint:
                #     best_val_loss = losses["val"]
                #     if iter_num > 0:
                #         checkpoint = {
                #             "model": raw_model.state_dict(),
                #             "optimizer": optimizer.state_dict(),
                #             "model_args": model_args,
                #             "iter_num": iter_num,
                #             "best_val_loss": best_val_loss,
                #             "config": config,
                #         }
                #         print(f"saving checkpoint to {out_dir}")
                #         torch.save(checkpoint, os.path.join(out_dir, "ckpt_sweep.pt"))
            if iter_num == 0 and config.eval_only:
                break

            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for micro_step in range(config.gradient_accumulation_steps):
                with ctx:
                    logits = model(X, Y)
                    loss = raw_model.last_loss
                    loss = loss / config.gradient_accumulation_steps # scale the loss to account for gradient accumulation
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                X, Y = get_batch('train', train_datasets, val_datasets, vocab, config, rng, device_type=device_type, bpe_tokenizer=bpe_tokenizer)
                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()
            # clip the gradient
            if config.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % config.log_interval == 0:
                # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
                lossf = loss.item() * config.gradient_accumulation_steps
                if local_iter_num >= 5:  # let the training loop settle a bit
                    mfu = raw_model.estimate_mfu(config.batch_size * config.gradient_accumulation_steps, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                print(
                    f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt*1000:.2f}ms | mfu {running_mfu*100:.2f}%"
                )
            iter_num += 1
            local_iter_num += 1

            # termination conditions
            if iter_num > config.max_iters:
                break


def main(config_file, count):
    with open(config_file) as f:
        config = yaml.safe_load(f)
    print("Sweep configuration:", config)
    print("Starting sweep...")
    sweep_id = wandb.sweep(config, project="MarketSimT_fast")
    wandb.agent(sweep_id, function=train_model, count=count)
    wandb.finish()
    print("Sweep completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to config file")
    parser.add_argument("--count", type=int, help="Number of runs to execute")
    args = parser.parse_args()
    main(args.config_file, args.count)