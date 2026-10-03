"""
This code was ported from: https://github.com/alxndrTL/mamba.py

Universal language model, which accepts as its core a Transformer or a Mamba.

The Transformer is implemented in PyTorch and supports FlashAttention-2/
For Mamba, you have the choice : use mamba.py's pure PyTorch implementation (cf mamba/mamba.py) or use the CUDA implementation.
"""

from typing import Union
import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# from ssm.transformer.transformer import Transformer, TransformerConfig, RMSNorm
from ssm.mamba import Mamba, MambaConfig, RMSNorm
# from mamba import Mamba, MambaConfig, RMSNorm
from jamba import Jamba, JambaConfig

# todo : inference function, with no grad, with kv cache for transformer, step() for mamba (see mamba.py/jamba.py)

class LM(nn.Module):
    def __init__(self, model_config: Union[MambaConfig, JambaConfig], vocab_size: int):
        super().__init__()

        self.config = model_config

        # self.embedding = nn.Embedding(vocab_size, self.config.d_model, padding_idx=0)
        self.embedding = nn.Embedding(vocab_size, self.config.d_model)
        
        # if isinstance(self.config, TransformerConfig):
        #     self.core = Transformer(self.config)
        # elif isinstance(self.config, MambaConfig):
        #     self.core = Mamba(self.config)
        if isinstance(self.config, MambaConfig):
            self.core = Mamba(self.config)
        elif isinstance(self.config, JambaConfig):
            self.core = Jamba(self.config)
        else:
            raise NotImplementedError

        self.out_norm = RMSNorm(self.config.d_model, self.config.rms_norm_eps)

        self.lm_head = nn.Linear(self.config.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('fc_3.weight') or pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.config.n_layers))

    def forward(self, tokens):
        # tokens : (B, L)

        # logits : (B, L, vocab_size)

        x = self.embedding(tokens)
        x = self.core(x)
        x = self.out_norm(x)
        logits = self.lm_head(x)

        return logits
    
    # taken from llama2.c
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # taken from llama2.c
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # any parameters that is 2D will be weight decayed, otherwise no. (i.e. all weight tensors in matmuls + embeddings decay, all biases and rmnsnorms don't)
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer
    
    def step(self, token, caches):
        # token : (B)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        # logits : (B, vocab_size)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        x = self.embedding(token)

        x, caches = self.core.step(x, caches)
        x = self.out_norm(x)

        logits = self.lm_head(x)

        return logits, caches
    
    # # TODO process prompt in parallel, and pass in sequential mode when prompt is finished ?
    # def generate2(self, tokenizer, prompt: str, num_tokens: int = 50, batch_size: int = 1, sample: bool = True, top_k: int = 40, temperature: float = 1.0):
    #     self.eval()

    #     input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(next(self.parameters()).device) # (1, num_tokens)
    #     input_ids = input_ids.repeat(batch_size, 1)

    #     # caches is a list of cache, one per layer
    #     # cache is composed of : the hidden state, and the last d_conv-1 inputs
    #     # the hidden state because the update is like an RNN
    #     # the last d_conv-1 inputs because they are used in a 1d convolution (usually d_conv=4 so this is not large)
    #     caches = [(None, torch.zeros(batch_size, self.config.d_inner, self.config.d_conv-1, device=input_ids.device)) for _ in range(self.config.n_layers)]

    #     for i in range(input_ids.size(1) + num_tokens - 1):
    #         with torch.no_grad():
    #             # forward the new output, get new cache
    #             next_token_logits, caches = self.step(input_ids[:, i], caches) # (batch_size, vocab_size), caches

    #         # sample (no sampling when the prompt is being processed)
    #         if i+1 >= input_ids.size(1):
    #             probs = F.softmax(next_token_logits / temperature, dim=-1) # (batch_size, vocab_size)

    #             if top_k is not None:
    #                 values, _ = torch.topk(probs, k=top_k) # (batch_size, k) ordered from lowest to biggest
    #                 probs[probs < values[:, -1, None]] = 0
    #                 probs = probs / probs.sum(axis=1, keepdims=True)

    #             if sample:
    #                 next_token = torch.multinomial(probs, num_samples=1).squeeze(1) # (batch_size)
    #             else:
    #                 next_token = torch.argmax(probs, dim=-1) # (batch_size)

    #             input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
                
    #     outputs = [tokenizer.decode(output.tolist()) for output in input_ids]

    #     self.train()

    #     if batch_size==1:
    #         return outputs[0]
    #     else:
    #         return outputs

    def relevant_mask(self, tok_pos, logits, vocab_encoding, device):
        # determine field and special tokens to include
        include_nan = False
        if tok_pos == 0:
            field = 'ticker'
        elif tok_pos == 1:
            field = 'type'
        elif tok_pos == 2:
            field = 'side'
        elif tok_pos in [3, 16]:
            field = 'sign'
            include_nan = True
        elif tok_pos in [4, 17]:
            field = 'price'
            include_nan = True
        elif tok_pos in [5, 6, 18]:
            field = 'size'
            include_nan = True
        elif tok_pos in [7, 8, 9, 10, 11, 12, 13, 14, 15]:
            field = 'time'
        elif tok_pos in [19, 20, 21, 22, 23]:
            field = 'time'
            include_nan = True

        # select relevant tokens, excluding special tokens
        if not include_nan:
            relevant_indices = vocab_encoding[field][1][3:]
        else:
            relevant_indices = vocab_encoding[field][1][2:]
        # prepare tensors for scattering
        relevant_destination = torch.zeros(1, logits.shape[1], device=device)
        relevant_indices = torch.tensor(relevant_indices, dtype=torch.int64, device=device).unsqueeze(0)
        relevant_src = torch.ones(1, relevant_indices.shape[1], device=device)
        # place the values in relevant_src at the indices in relevant_indices in relevant_destination
        relevant_mask = relevant_destination.scatter(1, index=relevant_indices, src=relevant_src).bool()
        # mask the indices in logits that are not relevant for the current field
        logits.masked_fill_(~relevant_mask, float("-inf"))
        return logits
    
    @torch.inference_mode()
    def generate(self, idx, caches, max_new_tokens, temperature=1.0, top_k=None, top_p = 0.9, start=False, vocab_encoding=None, use_relevant_mask=True):
    # def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p = 0.9, start=False, roll=False, new_block_size=10344, vocab_encoding=None, use_relevant_mask=True):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            idx: (b,t) LongTensor of input tokens
            max_new_tokens: int, the number of tokens to generate
            temperature >0: scale logits before applying softmax
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            start: bool, if True, the input is the start of a new sequence, otherwise it is a continuation
            roll: bool, if True, the input is the start of a rolling sequence and the KV cache is rolled
        """
        B,T = idx.shape
        for tok_pos in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            # idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
            # forward the model to get the logits for the index in the sequence
            if not start:
                x = idx[:,-1] #.reshape(-1,1)
            else:
                # pre-compute cache with full context
                x = idx
                for i in range(x.size(1)-1):
                    _, caches = self.step(x[:, i], caches)
                x = x[:, -1] #.reshape(-1,1)
            # sample (now that prompt has been processed)
            logits, caches = self.step(x, caches)
            if temperature == 0.0: # greedy sampling:
                # "sample" the single most likely index
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # pluck the logits at the final step and scale by desired temperature
                logits = logits / temperature
                # optionally mask indices that are not relevant for the current token field
                if use_relevant_mask:
                    logits = self.relevant_mask(tok_pos, logits, vocab_encoding, logits.device)
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                if top_p > 0.0:
                    # First sort and calculate cumulative sum of probabilities.
                    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
                    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
                    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
                    # scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits.masked_fill_(indices_to_remove, float("-inf"))
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            start = False
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx