import unittest,types
import numpy as np
from .joint_boundary import intercept

class CompleteJointBoundaryTests(unittest.TestCase):
    def setup(self,handler):
        names=['fl_hipx_joint']+['joint%d'%i for i in range(1,9)]+['fl_wheel_joint']
        c=types.SimpleNamespace(names=names,index={n:i for i,n in enumerate(names)},physics_step=10,
             articulation={'lower':np.full(10,-3.),'upper':np.full(10,3.)},
             attack_boundary=types.SimpleNamespace(intercept_joint_control=handler))
        r={'PHYSICS_DT':.005,'LEG_JOINTS':['fl_hipx_joint'],'M20_REVOLUTE_TARGET_LIMIT_MARGIN_RAD':.01,
           'VELOCITY_LIMIT_BY_NAME':{'fl_wheel_joint':2.}}
        q=np.array([.3,.1,.2,.3,.4,.5,.6,.03,-.03,np.nan],np.float32)
        qd=np.full(10,np.nan,np.float32);qd[-1]=.7
        return c,q,qd,r
    def test_clean_complete_transaction_is_numerically_identical(self):
        seen=[]
        def handler(v,**kw):seen.append(v);return v
        c,q,qd,r=self.setup(handler);a,b=intercept(c,q,qd,r)
        np.testing.assert_array_equal(q,a);np.testing.assert_array_equal(qd,b)
        self.assertEqual(len(seen),1);self.assertIn('joint6',seen[0]['positions'])
        self.assertIn('fl_hipx_joint',seen[0]['positions'])
    def test_arm_attack_does_not_change_legs_or_gripper(self):
        def handler(v,**kw):
            for i in range(1,7):v['positions']['joint%d'%i]=-.1*i
            return v
        c,q,qd,r=self.setup(handler);a,b=intercept(c,q,qd,r)
        np.testing.assert_allclose(a[1:7],np.arange(-.1,-.7,-.1),atol=1e-7)
        np.testing.assert_array_equal(a[[0,7,8]],q[[0,7,8]]);np.testing.assert_array_equal(b,qd)
    def test_original_basic_actuator_limits_remain_active(self):
        def handler(v,**kw):v['positions']['joint1']=100.;v['wheel_velocities']['fl_wheel_joint']=100.;return v
        c,q,qd,r=self.setup(handler);a,b=intercept(c,q,qd,r)
        self.assertEqual(a[1],3.);self.assertEqual(b[-1],2.)

if __name__=='__main__':unittest.main()
