import argparse
from tqdm import tqdm
from glob import glob
import os
import requests
import json
import numpy as np
import torch
import torch.distributed as dist
import random
from tokenizer import Tokeniser

from concurrent.futures import ThreadPoolExecutor



def download_file(url, fname, chunk_size = 1024):
    res = requests.get(url, stream = True)
    total = int(res.headers.get('content-length', 0))
    with open(fname, "wb") as f, tqdm(
        desc = fname,
        total = total,
        unit = "iB",
        unit_scale = True,
        unit_divisor = 1024
    ) as bar:
        for data in res.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)
    
DATA_CACHE_DIR = 'data'

def download():
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    data_filename = os.path.join(DATA_CACHE_DIR, "TinyStories.tar.gz")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download")
    
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok = True)
        print(f"Unpacking {data_filename}...")
        os.system(f"tar -xvf {data_filename} -C {data_dir}")
    else:
        print(f"{data_dir} already exists, skipping unpacking")
    
    shard_filenames = sorted(glob(os.path.join(data_dir, "*.json")))
    with open(shard_filenames[0], 'r') as f:
        data = json.load(f)
    print("Done")
    print(f"Number of shards: {len(shard_filenames)}")
    print(f"Example Story:\n {data[0]}")

def pretokenize():
    enc = Tokeniser()

    def process_shard(shard):
        with open(shard, 'r') as f:
            data = json.load(f)
        all_tokens = []
        for example in tqdm(data):
            text = example['story']
            text = text.strip()
            tokens = enc.encode(text, bos = True, eos = False)
            all_tokens.extend(tokens)
        
        all_tokens = np.array(all_tokens, dtype = np.uint16)

        tokenised_filename = shard.replace(".json", ".bin")
        with open(tokenised_filename, "wb") as b:
            b.write(all_tokens.tobytes())
        print(f"Saved {tokenised_filename}")

    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob(os.path.join(data_dir, '*.json')))
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(process_shard, shard_filenames)
    
    print("Done.")

class PreTokDataset(torch.utils.data.IterableDataset):
    def __init__(self, split, max_seq_len):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0

        rank = dist.get_rank() if dist.is_initialized() else 0

        seed = 42 + worker_id + 1337 + rank
        rng = random.Random(seed)
        print(f"Created a PreTokDataset with rng seed {seed}")
        data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
        shard_filenames = sorted(glob(os.path.join(data_dir, "*.bin")))

        shard_filenames = shard_filenames[1:] if self.split == 'train' else shard_filenames[:1]
        while True:
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                m = np.memmap(shard, dtype = np.uint16, mode = 'r')
                num_batches = len(m) // self.max_seq_len
                num_batches -= 1
                assert num_batches > 0, "this shard is way to small ?"
                ixs = list(range(num_batches))
                for ix in ixs:
                    start = ix * self.max_seq_len
                    end = start + self.max_seq_len + 1

                    chunk = torch.from_numpy((m[start: end].astype(np.int64)))
                    x = chunk[:-1]
                    y = chunk[1:]
                    yield x, y

class Task:
    @staticmethod
    def iter_batches(split, batch_size, max_seq_len, device, num_workers = 0):
        ds = PreTokDataset(split, max_seq_len)
        dl = torch.utils.data.DataLoader(
            ds, batch_size = batch_size, pin_memory = True, num_workers = num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking = True)
            y = y.to(device, non_blocking = True)
            yield x, y
        
