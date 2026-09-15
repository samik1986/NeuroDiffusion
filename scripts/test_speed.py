import time
from data_loader import NeuroDiffusionDataset, SWCParser
import os

file = "/mnt/diskg9-3/NeuroGramLM/SWCs/00/00000.swc"

t0 = time.time()
nodes, edges = SWCParser(file).parse(ignore_soma=True)
t_parse = time.time()
print(f"Parsed {file} in {t_parse - t0:.2f}s, nodes: {len(nodes)}")
