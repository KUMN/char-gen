from transformers import GPT2LMHeadModel, pipeline, set_seed
import matplotlib.pyplot as plt
import numpy as np
import torch
from dataclasses import dataclass
import torch.nn as nn
import math
import tiktoken
import inspect
import os
import time

# Creating model with same schema as seen in gpt2 model's state_dict


#--------------------------------without flash attention---------------------------

class CausalSelfAttentionNoFlash(nn.Module):
    # Multihead self attention - is a communication / weighted sum / aggregate / pooling layer
    # it is collection info across all tokens in the context - can be seen as a parallel reduce
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query. value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # adding custom attribute
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Masking the future context. It is not really a 'bias', but following OpenAI/HF naming
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1, 1, config.block_size, config.block_size))


    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # In gpt2 (124M) n_head=12, hs=64, nh*hs=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # attention (materializes the large (T, T) matrix for all the queries and keys)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # token affinity matrix
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf')) # only look at past tokens
        att = nn.functional.softmax(att, dim=-1) # normalizes to sum to 1
        y = att @ v # (B. nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs) # weighted sum of value based on attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y) # joining the residual path
        return y


#--------------------------------with flash attention----------------------------
#-----------------------------------------------------------

class CausalSelfAttention(nn.Module):
    # Multihead self attention - is a communication / weighted sum / aggregate / pooling layer
    # it is collection info across all tokens in the context - can be seen as a parallel reduce
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query. value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # adding custom attribute
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Masking the future context. It is not really a 'bias', but following OpenAI/HF naming
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1, 1, config.block_size, config.block_size))


    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # In gpt2 (124M) n_head=12, hs=64, nh*hs=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # flash attention does not materializes the large (T, T) matrix for all the queries and keys
        y = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # token affinity matrix
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf')) # only look at past tokens
        # att = nn.functional.softmax(att, dim=-1) # normalizes to sum to 1
        # y = att @ v # (B. nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs) # weighted sum of value based on attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y) # joining the residual path
        return y

class MLP(nn.Module):
    # Feed forward Layer - same weights or operation is applied to all tokens in the context
    # parallel map operation
    # it consists of two linear layers with an activation sandwiched
    # Activation is Gaussian Error Linear Unit, always has local gradient, smooth curve around zero
    # picked by BERT, LMs (Rectified Linear Unit has no gradient passage for values < 0)
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd) # widen the network
        # tanh approximated as earlier version was slow.
        # Today approximate version need not be in use, but using to mimic gpt2
        self.gelu = nn.GELU(approximate='tanh') # GELU(x) = x * \phi(x) phi is Gaussian
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd) # narrow back to n_embd
        self.c_proj.NANOGPT_SCALE_INIT = 1 # custom attribute

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x) # joining the residual path
        return x


class Block(nn.Module):
    # Attention and Feed Forward modules are defined in a block
    # Pernorm is called prior to attention and feedforward vs in Attention it is after
    # In Attention paper the residual connection was added and LayerNorm was called after it
    # this is not desireable where Layer Norm is in residual path. Ideally we want residual to
    # be straight clean path with only + operation from supervision to input.
    # When the gradient flows backward through '+'it gets distributed equally
    # through the residual path and the block path
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # prenorm -> attention -> join residual path
        x = x + self.mlp(self.ln_2(x)) # prenorm -> ffwd -> join residual path
        return x




@dataclass
class GPTConfig:
    block_size: int = 1024 #256 # context length, max sequence length
    # number of total tokens, 50K BPE merges, 256 byte tokens, 1 <|endoftext|>
    vocab_size: int = 50257 # 65 # vocabulary
    n_layer: int = 12 # 6 # number of transformer blocks
    n_head: int = 12 # 6 # number of multi-heads in attention
    n_embd: int = 768 # 384 # number of embedding


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        # Module Dict is a container that allows us to refer to the sub modules with keys
        # name the container transformer, submodules wte, wpe, h as listed in gpt2 schema
        # Module list is a container that allows us to address its submodules using int
        # Module list has n_layer number of Blocks - custom nn.Module containing attention, ffnn
        # ln_f is the prenorm before the final classifier lm_head
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # weight sharing scheme
        # Sharing weights between the embedding layer and the lm head; as the embeddings learned (putting in the inductive bias)
        # at both those ends should be similar or same, if we are to match the next best token by softmax semantic similarity
        # Refer to Section 3.4 in Attention is all you need paper - weight tieing is implemented. It is adopted in gpt2 as well
        # the weights at the lm_head also behave like word embeddings.
        # We get much better performance by learning during backward pass at each end (twice in a single pass),
        # as well as lesser model params in total (this is ~40M params in a 128M model)

        # init params
        self.apply(self._init_weights) # iterates through all the submodules of this module and applies functon init_weights on it


    # Initializing weights per gpt2 code for attention layer, biases, embedding and position embedding
    # Layer pre-norms: scale for layer norm = 1 and bias = 0 by default; we keep them

    def _init_weights(self, module):
      if isinstance(module, nn.Linear):
        # Linear layers in attention use normal distribution with stddev = 0.02
        # this std dev is ~ to sqrt(fan_in dimension) considering 768 dim for gpt2 or 1024, 1280, 1600 for other versions in gpt2
        std = 0.02
        # Section 2.3 in GP2 paper states that the residual layer weights at initialization are scaled by factor of 1/sqrt(N)
        # Each block as it joins the residual pathway at MLP and at Attention adds to the variance.
        # Each residual layer weight is scaled down to control the growth at the Nth residual layer the std dev remains ~1.0
        if hasattr(module, 'NANOGPT_SCALE_INIT'):
          std *= (2*self.config.n_layer) ** -0.5
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        # bias is initialized to 0. Pytorch has a small stdev for bias by default
        if module.bias is not None:
          torch.nn.init.zeros_(module.bias)
      # token embeddings have stddev = 0.02 and position embedding at 0.01, we use 0.02 for both
      elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets = None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size in"
        # forward the token and position embedding
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embedding of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embedding of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the block of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
          # calculate loss
          loss = nn.functional.cross_entropy(logits.view(B*T, self.config.vocab_size), targets.view(B*T))
        return logits, loss


    # custom optimizer configurer, to regularize weights
    # Two dimensional weights of Linear layers, embeddings needs to be regularized
    # It pulls down the weights so that just a few of them do not override others
    # All inputs / activations get to participate or contribute
    # bias need not be regularized, scale, shift - single dim params should not be regularized
    # AdamW fused option allows to run the optimization / update in a fuseed kernel for all parameters
    # it does not spin a separate kernel for each parameter tensors optimization
    def configure_optimizers(self, weight_decay, learning_rate, device):
      # start with all of the candidate parameters (that require grad)
      param_dict = {pn: p for pn, p in self.named_parameters()}
      param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
      # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
      # that is, all weight tensors in matmuls + embeddings decay, all biasea and layernorms don't
      decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
      nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
      optim_groups = [
          {'params': decay_params, 'weight_decay': weight_decay},
          {'params': nodecay_params, 'weight_decay': 0.0}
      ]
      num_decay_params = sum(p.numel() for p in decay_params)
      num_nodecay_params = sum(p.numel() for p in nodecay_params)
      print(f"num decayed parameter tensor: {len(decay_params)}, with {num_decay_params:,} param")
      print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} param")
      # Create AdamW optimizer and use the fused version if it is available
      fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
      use_fused = fused_available and 'cuda' in device
      print(f"using fused AdamW: {use_fused}")
      optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
      return optimizer

    # code to load weights from gpt2 model to our model and generate a sample
    # this is just to validate that the schema we produced is identical to that of gpt2
    # our model will be intialized with random weights and trained after this check
    # class methods is like constructor gets called when object is created
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained gpt2 model weights from hugging face"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':        dict(n_layer=12, n_head=12, n_embd=768), # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':  dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600) # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard thhis mask / buffer they are not weights

        # init a huggingface / transformer mdeol
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkppoints use a Conv1D module but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

#----------------------------DataLoader for training ----------------------------
class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B # batch size
        self.T = T # sequence length or token context length
        self.current_position = 0 # state advances by B * T after every batch

        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
          text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens") # loaded 338025 tokens 3:1 character to token ratio in gpt2
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches") # 1 epoch = 2640 batches



    def next_batch(self):
        B, T = self.B, self.T
        # take batch size +1 character
        data_buffer = self.tokens[self.current_position : self.current_position+((B*T)+1)]
        # advance the position in the tensor only by B * T (not B * T + 1)
        self.current_position += (B * T)
        # if loading the next batch would be out of bounds reset
        if self.current_position + ((B * T) + 1) > len(self.tokens):
          self.current_position = 0
        x = data_buffer[:-1].view(B, T) # inputs: take batch size, not the last character
        y = data_buffer[1:].view(B, T) # target labels: skip first character and take reset upto last one
        return x, y

# for distributed data parallel training
# Each process should run on their own set of data and GPUs should each get different batch copy
# This will run fine for single device where process_rank = 0 and num_processes = 1
class DataLoaderLiteMultiDevice:
    def __init__(self, B, T, process_rank, num_processes):
        self.B = B # batch size
        self.T = T # sequence length or token context length
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.current_position = self.B * self.T * self.process_rank # state advances by B * T * num_processes after every batch


        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
          text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens") # loaded 338025 tokens 3:1 character to token ratio in gpt2
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches") # 1 epoch = 2640 batches



    def next_batch(self):
        B, T = self.B, self.T
        # take batch size +1 character
        data_buffer = self.tokens[self.current_position : self.current_position+((B*T)+1)]
        # advance the position in the tensor only by B * T * self.num_processes (not B * T * self.num_processes + 1)
        self.current_position += (B * T * self.num_processes)
        # if loading the next batch would be out of bounds reset
        if self.current_position + ((B * T * self.num_processes) + 1) > len(self.tokens):
          self.current_position = self.B * self.T * self.process_rank
        x = data_buffer[:-1].view(B, T) # inputs: take batch size, not the last character
        y = data_buffer[1:].view(B, T) # target labels: skip first character and take reset upto last one
        return x, y


#----------------------multi gpu run-----------------------------------
 # simple launch: python file.py
 # DDP launch for 2 GPUs: torchrun --standalone --nproc_per_node=2 file.py

# -----------------------------------------------------------------------
# run the training loop
# call init_process_group with appropriate backend
# call destroy process group before exiting the process
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# setup DDP (distributed data parallel)
# torchrun command sets the env variables RANK, LOCAL_RANK and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
  # use of DDP at the moment for CUDA, we set the device appropriately according to rank
  # DDP support CPUs - multinode CPU or multisocket CPU training, usses Gloo backend
  assert torch.cuda.is_available(), "for now we need CUDA for ddp"
  init_process_group(backend='nccl')
  ddp_rank = int(os.environ['RANK']) # ranks 0 to World_size-1
  ddp_local_rank = int(os.environ['LOCAL_RANK']) # in a multi-node 0 to GPUs per node -1
  ddp_world_size = int(os.environ['WORLD_SIZE']) # Number of Nodes×GPUs per Node
  device = f'cuda:{ddp_local_rank}'
  torch.cuda.set_device(device)
  master_process = ddp_rank == 0 # if first GPU this process will do logging, checkpointing etc
else:
  # vanilla, non-DDP run
  ddp_rank = 0
  ddp_local_rank = 0
  ddp_world_size = 1
  master_process = True
  # attempt to autodetect device
  device = "cpu"
  # is GPU available
  if torch.cuda.is_available():
    device = "cuda"
  # check if Apple silicon is available, it has a powerful GPU as well
  elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
  print(f"using device: {device}")

# for reproducibility.
# Each parallel process will setup the model
# and initialize it with similar weights to begin with as all use same seed
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524288 # 2**19 ~0.5M in number of tokens
B = 4 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process: # else each process prints
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")


# create minibatch such that over the accumulation steps it totals to desired batchsize
# 1024 tokens per sample. 0.5M token in total in each batch means 0.5/1024 is ~488 samples
train_loader = DataLoaderLiteMultiDevice(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size) # pass rank and size
torch.set_float32_matmul_precision('high') # using tensorcore for A100 Nvidia series. This does not work for Tesla T4
# Random weights initialized in model - getting ready to train our own SLM
model = GPT(GPTConfig(vocab_size=50304)) # default config setting
model.to(device)
model = torch.compile(model) # compiled model - not interpreted
# wrap the model in DistributedDataParallel container
# the DDP proceeds with forward pass as before
# In backward pass after the loss.backward is done in each GPU
# it will call allreduce on the gradients to average them
# pass them back to each rank registered with the DDP
if ddp:
  model = DDP(model, device_ids=[ddp_local_rank]) # to GPU in the node
raw_model = model.module if ddp else model # raw unwrapped model

##### Setting Adam hyperparameters
max_lr = 6e-4 # referring to GPT3 small in table 2.1
min_lr = max_lr * 0.1 # 10% of max per paper
warmup_steps = 10
max_steps = 50
# defining the learning rate scheduler (pytorch has functions for it - better to custom implement)
def get_lr(it):
  # 1) Linear warmup for warmup iteration steps
  if it < warmup_steps:
    return max_lr * ((it + 1) / warmup_steps)
  # 2) if it > lr_decay_iters, return min learning rate
  if it > max_steps:
    return min_lr
  # 3) In between, use cosine decay down to min learning rate
  decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
  assert 0 <= decay_ratio <= 1
  coeff = 0.5 *(1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
  return min_lr + coeff * (max_lr - min_lr)


# Adam is optimizer like SGD but converges faster to optimal weights
# - has first moment (momentum like RMSProp) applied to gradients and second moment applied to gradient square (non-centred variance)
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8) # AdamW bug fix version of Adam
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

# now the model will not overfit
# most of the easy gains at this point are by push down the weights of gpt2 tokens,
# that never occur in this training set to zero
for step in range(max_steps):
  t0 = time.time()
  optimizer.zero_grad(set_to_none=True) # clear the gradient
  ## Gradient accumulation
  loss_accum = 0.0
  for micro_step in range(grad_accum_steps):
      x, y = train_loader.next_batch()
      x, y = x.to(device), y.to(device)
      # the matrix multiply in the lm_head layer is about 30%
      # it dominates matrix multiply among all the other layers in the architecture
      with torch.autocast(device_type=device, dtype=torch.bfloat16):
          logits, loss = model(x, y) # call model and calculate loss for one pass
          #import code; code.interact(local=locals()) # ctrl D to continue
      loss = loss / grad_accum_steps # divide by accumulation steps - normalizer
      loss_accum += loss.detach() # detaching tensor from graph # this is the local ranks loss not ddp average loss
      # DDP should not kickin at every microstep loss.backward to publish the gradients from this rank
      # Synchronize should happen only at the last microstep of this step
      # and call allreduce to average the gradients
      # model has require_backward_grad_sync flag that can be toggled to set synchronization

      #Official sanctioned way to stop sync by ddp is to use no_sync()
      #with ddp.no_sync():
      #    for input in inputs:
      #        ddp(input).backward() # no synchronization, accumulate grads
      #ddp(another_input).backward() # synchronization of grads
      if ddp:
        model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        # at the point this sets to True ddp will synchronize, all reduce and pass the average to all ranks
      loss.backward() # accumulate the gradients
  # Reduces the tensor data across all machines in a way that all get the final result.
  # After the call tensor is going to be bitwise identical in all processes
  if ddp: # collects loss_accum from each rank and publishes its average to all ranks
    dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

  norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # clip the gradients based on norm
  # determine and set the learning rate for this iteration
  lr = get_lr(step)
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  optimizer.step() # update the model parameters
  # torch.cuda.synchronize() # waits for work assigned to GPU cores to complete
  t1 = time.time()
  dt = (t1 - t0)* 1000 # time difference in milliseconds
  tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
  tokens_per_sec  = tokens_processed / dt
  if master_process:
      print(f"step {step} | loss: {loss_accum.item()}| lr {lr:.4e} | norm: {norm:.4f}| dt {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}") # ships the loss tensor to cpu and picks the item scalar value


print("final loss", loss) # expected loss is -ln(1/50257) ~10.8
print(logits.shape) # (B, T, vocab_size)
if ddp:
  destroy_process_group() # called when ddp exits each process