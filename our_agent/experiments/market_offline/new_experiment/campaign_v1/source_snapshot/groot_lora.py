import torch
from torch import nn
class LowRank(nn.Module):
 def __init__(self,base,rank=32):
  super().__init__();self.base=base
  i,o=(base.in_features,base.out_features) if isinstance(base,nn.Linear) else base.W.shape[1:]
  self.lora_A=nn.Linear(i,rank,bias=False);self.lora_B=nn.Linear(rank,o,bias=False)
  nn.init.normal_(self.lora_A.weight,std=.02);nn.init.zeros_(self.lora_B.weight)
 @property
 def weight(self):return self.base.weight
 @property
 def bias(self):return self.base.bias
 def forward(self,x,*args,**kwargs):
  orig=self.base(x,*args,**kwargs);delta=self.lora_B(self.lora_A(x.to(self.lora_A.weight.dtype)))
  return orig+delta.to(orig.dtype)
def install(model,rank=32):
 model.requires_grad_(False);model.to(dtype=torch.bfloat16);names=[]
 for name,module in list(model.named_modules()):
  if name.startswith('action_head.') and not any(x in name for x in ['vlln','vl_self_attention']) and (isinstance(module,nn.Linear) or type(module).__name__=='CategorySpecificLinear'):
   parent,leaf=name.rsplit('.',1);setattr(model.get_submodule(parent),leaf,LowRank(module,rank));names.append(name)
 assert names
 return {'type':'action_head_low_rank_adaptation','rank':rank,'alpha':rank,'modules':names,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}
