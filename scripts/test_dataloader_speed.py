import time
from data_loader import NeuroDiffusionDataset
from utils import load_config
import os
import cProfile
import pstats
import glob

config = load_config("config.yaml")

dataset = NeuroDiffusionDataset.__new__(NeuroDiffusionDataset)
dataset.data_dir = config['data']['swc_dir']
dataset.laplacian_k = 8
dataset.max_nodes = 1000
dataset.overlap = 200
dataset.swc_files = glob.glob(config['data']['swc_dir'] + "/**/*.swc", recursive=True)[:10]
dataset.split = 'train'

def profile_item():
    for i in range(10):
        dataset[i]

cProfile.run('profile_item()', 'profile_stats_item')
p = pstats.Stats('profile_stats_item')
p.sort_stats('cumulative').print_stats(20)
