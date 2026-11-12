import sys,types,unittest
import numpy as np
from .common import PROJECT
sys.path.insert(0,str(PROJECT/'datasets/paper200/collection_20260924_v2/runtime/runner/m20_runtime'))
from attack_tools.attack_orchestrator import AttackOrchestrator
from attack_tools.attackable_vla_client import AttackableVLAClient

class RequestBoundaryTests(unittest.TestCase):
 def setup(self):
  events=[];seen=[]
  cfg={'attacks':[{'id':'image','mode':'intercept','stage':'request_image','handler':'patch','duration_s':1.,'release_delay_s':0.,'trigger':{'all':[{'type':'field','field':'visible','op':'truthy'}]}}]}
  def patch(x,p,c):return x+1
  orch=AttackOrchestrator(cfg,{'patch':patch},audit=events.append)
  def infer(obs):seen.append(obs);return {'actions':np.zeros((1,10),np.float32)}
  recorder=types.SimpleNamespace(append=events.append,capture_vla_transaction=lambda *a,**kw:None,
      capture_consumption=events.append)
  client=AttackableVLAClient(types.SimpleNamespace(infer=infer),globally_enabled=True,orchestrator=orch,recorder=recorder,config_path=None)
  return client,events,seen
 def test_legacy_bootstrap_preserves_current_physical_condition(self):
  c,events,seen=self.setup();c.observe_attack_state({'visible':True,'simulation':{'time_s':0.,'physics_step':0}})
  img=np.zeros((3,3,3),np.uint8)
  c.infer({'observation/image':img,'observation/m20_camera_0_rgb':img},_attack_context={'time_s':0.,'physics_step':0})
  np.testing.assert_array_equal(seen[0]['observation/image'],img+1)
  np.testing.assert_array_equal(seen[0]['observation/m20_camera_0_rgb'],img+1)
  self.assertTrue(any(x.get('event')=='effect_first_applied' for x in events))
 def test_actual_region_exit_still_releases_attack(self):
  c,events,seen=self.setup();c.observe_attack_state({'visible':True,'simulation':{'time_s':0.,'physics_step':0}})
  c.observe_attack_state({'visible':False,'simulation':{'time_s':.1,'physics_step':20}})
  img=np.zeros((3,3,3),np.uint8);c.infer({'observation/image':img},_attack_context={'time_s':.1,'physics_step':20})
  np.testing.assert_array_equal(seen[0]['observation/image'],img)

if __name__=='__main__':unittest.main()
