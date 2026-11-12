import torch,sys,pathlib,importlib.util,json
from groot_lora import LowRank
R=pathlib.Path(__file__).resolve().parents[2];p=R/'restored/VLA/data/grasp_755/vendor/Isaac-GR00T/gr00t/model/modules/embodiment_conditioned_mlp.py';spec=importlib.util.spec_from_file_location('category',p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
rows=[]
for base,args in [(torch.nn.Linear(5,7).bfloat16(),()),(mod.CategorySpecificLinear(3,5,7).bfloat16(),(torch.tensor([1,2]),))]:
 base.requires_grad_(False);x=torch.randn(2,4,5,dtype=torch.bfloat16);ref=base(x,*args);adapter=LowRank(base);out=adapter(x,*args);assert torch.equal(out,ref);out.float().square().mean().backward();assert adapter.lora_B.weight.grad.abs().sum()>0;assert all(p.grad is None for p in base.parameters());rows.append({'base':type(base).__name__,'zero_adapter_exact_identity':True,'adapter_gradient_nonzero':True,'base_gradient_absent':True})
(R/'new_experiment/ADAPTER_TEST.json').write_text(json.dumps(rows,indent=2));print(rows)
