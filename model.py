import torch
import torch.nn as nn
import torch.nn.functional as F

from rotary_embedding_torch import RotaryEmbedding

nembed = 128
nheads=  4
nlayers = 4
block_size = 128
batch_size = 64

device = 'cuda'

gamma = (1 - 2 ** (-5 - torch.arange(nheads, dtype = torch.float32)))
range_tensor = torch.arange(block_size)
range_tensor = range_tensor[None, :, None].expand(nheads, block_size, 1)
diff_tensor = range_tensor - range_tensor.transpose(-1, -2)
decays = gamma[:, None, None] ** diff_tensor
mask = torch.tril(decays).to(device)


with open("input.txt", 'r', encoding = "utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encode = lambda x: [stoi[s] for s in x]
decode = lambda x: "".join([itos[i] for i in x])

data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size - 1, (batch_size, ))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class MultiScaleRetention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(nembed, 3 * nembed, bias = False)
        self.gated = nn.Linear(nembed, nembed, bias = False)
        self.proj = nn.Linear(nembed, nembed, bias = False)
        self.gn = nn.GroupNorm(num_groups = nheads, num_channels = nembed)

        self.rope = RotaryEmbedding(dim = nembed // nheads, use_xpos=True)

    def parallel_retention(self, q, k, v, T):
        retention = q @ k.transpose(-1, -2) * k.size(-1) ** -0.5
        retention = retention * mask[:, :T, :T]
        y = retention @ v
        return y

    def recurrent_retention(self, q, k, v, past_kv):
        pass

    def forward(self, x, past_kv):
        B, T, C = x.size()
        q, k, v = self.qkv(x).split(nembed, dim = -1)
        q = q.view(B, T, nheads, - 1).transpose(1, 2)
        k = k.view(B, T, nheads, - 1).transpose(1, 2)
        v = v.view(B, T, nheads, - 1).transpose(1, 2)

        q, k = self.rope.rotate_queries_and_keys(q, k)

        if past_kv == None: 
            y = self.parallel_retention(q, k, v, T)
            curr_kv = None
        else:
            y, curr_kv = self.recurrent_retention(q, k, v, past_kv)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.gn(y.transpose(1, 2)).transpose(1, 2)

        out = F.silu(self.gated(x)) * y
        return self.proj(out), curr_kv

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.msr = MultiScaleRetention()
        self.mlp = nn.Sequential(
            nn.Linear(nembed, 2 * nembed),
            nn.GELU(),
            nn.Linear(nembed * 2, nembed),
        )

        self.ln1 = nn.LayerNorm(nembed)
        self.ln2 = nn.LayerNorm(nembed)

    def forward(self, x, past_kv = None):
        msr, curr_kv = self.msr(self.ln1(x), past_kv)
        x = x + msr
        x = x + self.mlp(self.ln2(x))
        return x, curr_kv
    
class GPT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, nembed)
        # self.msr = MultiScaleRetention()
        self.blocks = nn.ModuleList([Block() for _ in range(nlayers)])
        self.proj = nn.Linear(nembed, vocab_size)
    
    def forward(self, idx, targets = None):
        B, T = idx.size()
        x = self.emb(idx)

        if self.training:
            # past_kv = torch.zeros_like(x).view(B, T, nheads, -1).transpose(1, 2)
            past_kv = None
        else:
            past_kv = None

        for layer in self.blocks:
            x, past_kv = layer(x, past_kv)

        logits = self.proj(x)
        if targets == None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, steps = 200):
        self.eval()
        for _ in range(steps):
            logits, _ = self(idx) if idx.size(-1) <= block_size else self(idx[:, -block_size:])
            logits = logits[:, -1]
            probs = F.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim = -1)
        self.train()
        return idx

model = GPT().to(device)
print("Model Parameters: ", sum(p.numel() for p in model.parameters()))

max_iters = 2000
eval_iters = 50
optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)

from tqdm import tqdm
pbar = tqdm(range(max_iters))

for i in pbar:
    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none = True)
    pbar.set_description(f"Loss {loss.item():.4f}")
    if i % 100== 0:
        losses = estimate_loss()
        print(f"\nStep {i}: train_loss {losses['train']:.4f}  val_loss {losses['val']:.4f}")


context = "\n"
out = model.generate(torch.tensor([encode(context)], dtype = torch.long, device = device), steps = 300)
print("".join(decode(out[0].tolist())))

