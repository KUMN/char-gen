import torch # Pytorch: https://pytorch.org
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
torch.manual_seed(1337) # for reproducibility
batch_size = 32 # number of independent sequences processed in parallel
block_size = 8 # maximum context length for predictions
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu' #use cuda / GPU if available
eval_iters = 200
#------------
# super simple bigram model that predicts next char based on input char
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off for the next token from a lookup table
        # since each row is vocab size it serves as the logits
        # parameters are moved to GPU if model is moved to device
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) 

    def forward(self, idx, target=None):
        #idx and targets are both (B, T) tensor of integers
        logits = self.token_embedding_table(idx) 
        B, T, C = logits.shape #(B,T,C) batch, block_size, embedding
        if target == None:
            loss = None
        else:
            # Cross Entropy - If dimensions > 2, pytorch expects embeddings at dim=1 --> B,C,T or B,C
            loss = F.cross_entropy(logits.view(B*T, C), target.view(B*T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        # get the prediction for T+1
        # pick the C for that last prediction, convert to probabilities and sample from it
        for _ in range(max_new_tokens):
            # get the predictions
            # self(idx) calls forward function, but Y is not passed update forward to take Y optionally
            logits, loss = self(idx) 
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C) # Get the C for prediction of last T (timestep)
            # apply softmax to get the probabilities
            probs = F.softmax(logits, dim=-1) # (B, C) # Convert logits to probabilities
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) #(B, 1) # Sample from prob distribution
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx



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
model = BigramLanguageModel(vocab_size)
m = model.to(device) # move model parameters essentially to cuda if it exists

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

# to generate a sample start token is idx=0 is '\n'
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("generated text: ", decode(m.generate(context, max_new_tokens=100)[0].tolist()))