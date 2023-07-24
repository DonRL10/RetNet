import torch 
import torch.nn as nn 
import torch.nn.functional as F

from xpos import XPOS

def get_decay_mask(nheads, block_size, device, type = "parallel"):
    gamma = 1 - 2 ** (-5 - torch.arange(nheads, dtype = torch.float32))
    if type == 'recurrent':
        return gamma[:, None, None].to(device)
    range_tensor = torch.arange(block_size)
    range_tensor = range_tensor[None, :, None].expand(nheads, block_size, 1)
    diff_tensor = range_tensor - range_tensor.transpose(-1, -2)
    mask = gamma[:, None, None] ** diff_tensor
    return torch.tril(mask).to(device)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.scale = dim ** -0.5

    def forward(self, x):
        out = torch.norm(x, dim = -1, keepdim = True) * self.scale
        return x / out.clamp(min = 1e-8) * self.weight


class Fused(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.fused_dims = (config.nembed, config.nembed, config.nembed, (config.nembed * 2))
        self.qkvff = nn.Linear(config.nembed, sum(self.fused_dims), bias = False)

        self.gated = nn.Linear(config.nembed, config.nembed)
        self.proj = nn.Linear(config.nembed, config.nembed)

        self.gn = nn.GroupNorm(config.nheads, config.nembed)

        self.ff_proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(config.nembed * 2, config.nembed, bias = False)
        )
        
        self.norm = RMSNorm(config.nembed)

        self.xpos = XPOS(config.nembed)

        self.nheads = config.nheads

    def parallel_retention(self, q, k, v, T):
        ret = q @ k.transpose(-1, -2)  * k.shape[-1] ** -0.5
        ret = ret * get_decay_mask(self.nheads, T, device = ret.device)
        y = ret @ v
        return y

    def recurrent_retention(self, q, k, v, T, past_kv):
        past_kv = 0 if past_kv == None else past_kv
        gamma = get_decay_mask(self.nheads, T, q.device, type = "recurrent")
        curr_kv = gamma * past_kv + k.transpose(-1, -2) @ v
        ret = q @ curr_kv * k.size(-1) ** -0.5
        return ret, curr_kv

    def forward(self, x, past_kv = None, type = 'parallel', offset = 0):
        B, T, C = x.size()
        x_ = self.norm(x)
        q, k, v, ff = self.qkvff(x_).split(self.fused_dims, dim = -1)

        q = self.xpos(q, offset = offset)
        k = self.xpos(k, offset = offset, downscale = True)

        q = q.view(B, T, self.nheads, -1).transpose(1, 2)
        v = v.view(B, T, self.nheads, -1).transpose(1, 2)
        k = k.view(B, T, self.nheads, -1).transpose(1, 2)

        if type == 'parallel':
            y = self.parallel_retention(q, k, v, T)
            curr_kv = None
        else:
            y, curr_kv = self.recurrent_retention(q, k, v, T, past_kv)
        
        y = y.transpose(1, 2).contiguous().view(-1, C)
        y = self.gn(y).view(B, T, C)
        y = F.silu(self.gated(x)) * y
        y = self.proj(y)

        ff_out = self.ff_proj(ff)

        return x + ff_out + y, curr_kv

class RetNET(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.emb = nn.Embedding(config.vocab_size, config.nembed)
        self.blocks = nn.ModuleList([Fused(config) for _ in range(config.nlayers)])

        self.rms = RMSNorm(config.nembed)
        self.proj = nn.Linear(config.nembed, config.vocab_size)

    def forward(self, idx, targets = None, past_kv = None, type = "parallel", offset = 0):
        x = self.emb(idx)

        kv_cache = []
        for i, layer in enumerate(self.blocks):
            past_kv_i = past_kv[i] if past_kv != None else None
            x, kv_i = layer(x, past_kv_i, type = type, offset = offset)
            kv_cache.append(kv_i)

        # x = self.rms(x)
        logits = self.proj(x)

        if targets == None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, kv_cache

    @torch.no_grad()
    def generate(self, idx, steps = 100):
        past_kv = None
        for i in range(idx.shape[-1]):
            _, _, past_kv = self(idx[:, i: i + 1], past_kv = past_kv, type = "recurrent", offset = i)
        for _ in range(steps):
            logits, _, past_kv = self(idx[:, [-1]], past_kv = past_kv, type = "recurrent", offset = idx.size(-1) - 1)
            probs = logits[:, -1].softmax(-1)
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat([idx, idx_next], dim = -1)
        return idx
    
