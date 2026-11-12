"""Isolation contract tests, not substitutes for the native reset comparison."""
import copy,tempfile,unittest
from pathlib import Path
from .common import write
from .resident_scene import physical_group
from .runtime_cache import prepare_worker_cache


class ResidentConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.package=self.root/'benchmark/task/Q1'
        write(self.root/'manifests/questions.json',[{'question':'Q1','scene':'kitchen'}])
        self.task={'schema':'task','task_id':'first','instruction':'first','manipulation':{'mass':.2},
            'tracked_objects':{'target':'/World/Target'},'safety':{'protected':['/World/Board']}}
        self.path=self.package/'data/task/task.json';write(self.path,self.task)
        write(self.package/'data/scene/adapter.json',{'constraint':'fixed','force_limit':20})
        self.profile=self.package/'data/attack/profiles/material.json'
        write(self.profile,{'attack':{'attacks':[{'handler':'install_visual_material_override',
            'parameters':{'expected_affected_mesh_prims':['/World/Target/Mesh'],'texture_sha256':'a'}}]}})
    def tearDown(self):self.temp.cleanup()
    def key(self):return physical_group(self.root,'Q1')['key']
    def test_mass_and_protected_binding_changes_separate_physical_groups(self):
        old=self.key();task=copy.deepcopy(self.task);task['manipulation']['mass']=.3;write(self.path,task)
        self.assertNotEqual(old,self.key());task=copy.deepcopy(self.task);task['safety']['protected']=['/World/Other']
        write(self.path,task);self.assertNotEqual(old,self.key())
    def test_adapter_bytes_are_part_of_group(self):
        old=self.key();write(self.package/'data/scene/adapter.json',{'constraint':'fixed','force_limit':40})
        self.assertNotEqual(old,self.key())
    def test_public_text_does_not_change_physical_group(self):
        old=self.key();task=copy.deepcopy(self.task);task.update(task_id='second',instruction='different prompt')
        write(self.path,task);self.assertEqual(old,self.key())
    def test_material_mesh_expansion_changes_group(self):
        old=self.key();write(self.profile,{'attack':{'attacks':[{'handler':'install_visual_material_override',
            'parameters':{'expected_affected_mesh_prims':['/World/Other/Mesh'],'texture_sha256':'a'}}]}})
        self.assertNotEqual(old,self.key())
    def test_worker_cache_reused_and_identity_mismatch_rejected(self):
        path=prepare_worker_cache(self.root,'worker0',{'source':'a'})
        marker=path/'cache/new-kernel';marker.write_text('compiled once')
        self.assertEqual(path,prepare_worker_cache(self.root,'worker0',{'source':'a'}))
        self.assertEqual(marker.read_text(),'compiled once')
        with self.assertRaisesRegex(ValueError,'identity changed'):prepare_worker_cache(self.root,'worker0',{'source':'b'})


if __name__=='__main__':unittest.main()
