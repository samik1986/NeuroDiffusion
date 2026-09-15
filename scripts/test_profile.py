import cProfile
import pstats
from data_loader import NeuroDiffusionDataset
import glob

dataset = NeuroDiffusionDataset("/mnt/diskg9-3/NeuroGramLM/SWCs")
file = glob.glob("/mnt/diskg9-3/NeuroGramLM/SWCs/**/*.swc", recursive=True)[0]
print(f"Profiling file: {file}")

def profile_file():
    dataset[0] # this will parse the file and do everything!
    
cProfile.run('profile_file()', 'profile_stats')
p = pstats.Stats('profile_stats')
p.sort_stats('cumulative').print_stats(20)
