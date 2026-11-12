import pathlib,sys,json,numpy as np
R=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'evidence/VLA/training'))
from market_dataset import EpisodeReader,SamplingPlan
class Reader(EpisodeReader):
 def __init__(self,split='train'):
  super().__init__(R/'restored/VLA/data/grasp_755')
  self.entries=[json.loads(l) for l in (R/'new_experiment'/f'{split}.jsonl').read_text().splitlines()]
class Data:
 def __init__(self,seed,split='train'):
  self.reader=Reader(split);self.plan=SamplingPlan(self.reader,seed=seed);self.stats=json.loads((R/'new_experiment/train_only_stats.json').read_text())
 def sample(self,i):return self.reader.read(*self.plan.choice(i))
