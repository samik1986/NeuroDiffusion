import time
from data_loader import get_dataloader
from utils import load_config
import os

config = load_config("config.yaml")
# Ensure we don't use the IterableDataset by temporarily hiding the chunks dir
chunk_dir = os.path.join(config['data']['swc_dir'], "processed_chunks")
if os.path.exists(chunk_dir):
    os.rename(chunk_dir, chunk_dir + "_hidden")

train_loader, _ = get_dataloader(config)

if os.path.exists(chunk_dir + "_hidden"):
    os.rename(chunk_dir + "_hidden", chunk_dir)

print(f"Dataloader created. Starting iteration...")
t0 = time.time()
num_batches = 0
for i, batch in enumerate(train_loader):
    if batch is None: continue
    num_batches += 1
    if i % 10 == 0:
        print(f"Batch {i} loaded in {time.time() - t0:.2f}s")
        t0 = time.time()
    if i == 50:
        break
print(f"Finished {num_batches} batches.")
