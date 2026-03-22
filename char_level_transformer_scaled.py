import torch # Pytorch: https://pytorch.org
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
torch.manual_seed(1337) # for reproducibility
batch_size = 64 # 32 # number of independent sequences processed in parallel
block_size = 256 # 8 # maximum context length for predictions
max_iters = 5000 # max iterations increased as learning rate is reduced
eval_interval = 500
learning_rate = 3e-4 # le-2 for simple model but attention needs lower learning rate
device = 'cuda' if torch.cuda.is_available() else 'cpu' #use cuda / GPU if available
eval_iters = 200
vocab_size = 0 # will be initialized after reading data
n_embd = 384 # 32
n_head = 6
head_size = n_embd // n_head
n_layers = 6 # number of transformer blocks
dropout = 0.2
#------------

# Character level decoder only model that predicts next char based on input char
class CharLevelLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off for the next token from a lookup table
        # since each row is vocab size it serves as the logits
        # parameters are moved to GPU if model is moved to device
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # to encode token
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # to encode position of token
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head=n_head) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(n_embd) #final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size) # output head
        
    
    def forward(self, idx, target=None):
        #idx and targets are both (B, T) tensor of integers
        B, T = idx.shape
        x_emb = self.token_embedding_table(idx) # x_emb are B, T, C
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # T, C
        x = x_emb + pos_emb # shape T, C broadcast to B
        # placing attention and feedforward inside block
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        B, T, vocab_size = logits.shape #(B,T,vocab_size) batch, block_size, embedding
        if target is None:
            loss = None
        else:
            # Cross Entropy - If dimensions > 2, pytorch expects embeddings at dim=1 --> B,vocab,T or B,vocab
            loss = F.cross_entropy(logits.view(B*T, vocab_size), target.view(B*T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        # get the prediction for T+1
        # pick the C for that last prediction, convert to probabilities and sample from it
        for _ in range(max_new_tokens):
            # get the predictions
            # crop idx to the last block_size tokens - it grows as it loops
            idx_cond = idx[:, -block_size:]
            # self(idx) calls forward function, but Y is not passed update forward to take Y optionally
            logits, loss = self(idx_cond) 
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C) # Get the C for prediction of last T (timestep)
            # apply softmax to get the probabilities
            probs = F.softmax(logits, dim=-1) # (B, C) # Convert logits to probabilities
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) #(B, 1) # Sample from prob distribution
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

class SingleHeadAttention(nn.Module):
    "one head self-attention"
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout) # dropout after softmax to regularize affinity matrix

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # B, T, head_size
        q = self.query(x) # B, T, head_size
        head_size = q.shape[-1]
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * head_size**-0.5 # (B, T, H) @ (B, H, T) ---> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, head_size)
        out = wei @ v # (B, T, T) @ (B, T, head_size) ---> (B, T, head_size)
        return out

class MultiHeadAttentionWithProjection(nn.Module):
    # multiple heads of self attention in parallel
    def __init__(self, num_heads, head_size):
        # n_embd: embedding dimension
        # n_head: number of heads
        super().__init__()
        self.heads = nn.ModuleList([SingleHeadAttention(head_size) for _ in range(num_heads)])
        #create projections of output to x sizes, to allow for additions of skip connections
        self.project = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # each head is run in parallel with the same input
        head_outputs = [h(x) for h in self.heads]
        out = torch.cat(head_outputs, dim=-1)
        out = self.project(out)
        out = self.dropout(out) # dropout before joining the residual connection
        # concatenate output from each head in the last dimension (head_size)
        # output size after concatenation is num_heads * head_size
        return out
        
# going directly from the multihead output to the language model head does not give the model a chance
# to think through or consolidate the outputs from multiple heads
# W = n_embd X n_embd example is a single set that is broadcast per token and applied in parallel
# Each token thinks through the data individually
class FeedForwardWithProjection(nn.Module):
    # a simple linear layer followed by a non-linearity
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd*4),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd), #projection layer
            nn.Dropout(dropout), # at the point of joining back the residual connection
        )

    def forward(self, x):
        out = self.net(x)
        return out

class TransformerBlock(nn.Module):
    # Transformer block, communication followed by computation
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension
        # n_head: number of heads
        super().__init__()
        head_size = n_embd // n_head
        self.self_attention = MultiHeadAttentionWithProjection(n_head, head_size)
        self.ffwd = FeedForwardWithProjection(n_embd)
        self.layernorm1 = nn.LayerNorm(n_embd) # Layer norm from Pytorch # MyLayerNorm1d(n_embd) 
        self.layernorm2 = nn.LayerNorm(n_embd) # Second LayerNorm after FFNN # MyLayerNorm1d(n_embd) 

    def forward(self, x):
        x = x + self.self_attention(self.layernorm1(x)) # prenorm/layernorm -> self-attention -> residual/skip connection 
        x = x + self.ffwd(self.layernorm2(x)) # prenorm / layernorm -> ffnn -> residual/skip connection
        return x

# Layer Norm Implemetation listed for illustration. In the model nn.LayerNorm Pytorch's implementation is used
class MyLayerNorm1d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        # parameters (trained with backprop)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        
    def forward(self, x):
         #def __call__(self, x):
        # calculate the forward pass
        xmean = x.mean(1, keepdim=True) # mean across the sample's dimensions (features)
        xvar = x.var(1, keepdim=True, correction=0) #, unbiased=True) # variance across the sample's features (dimension)
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
        self.out = self.gamma + xhat + self.beta
        return self.out
    

# read input file

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Tokenization: segment the input to atomic units/elements. For this model the units are character level
# All inputs to the model are sequences of tokens, represented by numerical indices of the vocabulary.
chars_input = sorted(set(text)) # create character level vocabulary
vocab_size = len(chars_input)
# map the vocabulary to integers
stoi, itos = {}, {}
for i, c in enumerate(chars_input):
    stoi[c] = i
    itos[i] = c
encode = lambda str_input: [stoi[c] for c in str_input] # encoder: input-string, output-sequence of int
decode = lambda ix_input: "".join([itos[i] for i in ix_input]) # decoder: input-sequence of int, output-string

# Encoding the entire text dataset and storing into a torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)
# splitting data into train and validation sets
n = int(0.9*len(data)) # first 90% will be train, rest validation
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate small batch of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(low = 0, high = len(data)-block_size, size=(batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device) # move data to cuda if it exists
    return x, y

# tell pytorch that backward will never be called on this function's computation graph 
# pytorch does not store the intermediate steps in anticipation of backward pass, be memory efficient 
@torch.no_grad() 
def estimate_loss():
    out = {}
    model.eval() # change model to evaluation mode. In layers BatchNorm, Dropout etc training is set to False
    for split in ['train', 'val']: # evaluate over both splits
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters): # evaluate over a few batches
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean() # use the mean over few batches to get stable estimates
    model.train() # set the mode back to training = True in the layers
    return out

# lets train the bigram model - this trained model will be the baseline
model = CharLevelLanguageModel() # vocab_size is a global variable
m = model.to(device) # move model parameters essentially to cuda if it exists
print(len(m.parameters()))
# optimization typically is done with SGD. Here we use AdamW optimizer - lr is typically 1e-4 or 1e-3
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate) # for small networks the lr can take higer values

for iter in range(max_iters):
    # every once in a while evaluate loss a few batches of training and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # sample a batch of training data
    xb, yb = get_batch('train')
    # evaluate the loss
    logits, loss = m(xb, yb) # forward pass
    optimizer.zero_grad(set_to_none=True) # flush the gradients
    loss.backward() # backpropagation
    optimizer.step() # update parameters by -lr*gradients

    break # to trial run

# to generate a sample start token is idx=0 is '\n'
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("generated text: ", decode(m.generate(context, max_new_tokens=100)[0].tolist()))