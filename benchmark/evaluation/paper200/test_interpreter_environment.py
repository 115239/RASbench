import json
import os
import subprocess
import unittest
from paper200.common import NATIVE
from paper200.launch import interpreter_environment


class InterpreterEnvironmentTests(unittest.TestCase):
    def test_native_entrypoint_overrides_inherited_kind_and_loads_real_dependencies(self):
        for desired, polluted, minor, dependency in [('model', 'isaac', 12, 'tyro'), ('isaac', 'model', 11, 'h5py')]:
            original=dict(os.environ,BASELINE_PYTHON_KIND=polluted)
            selected=interpreter_environment(original,desired)
            result=subprocess.run([str(NATIVE/'python.sh'),'-c',
                'import sys,json,%s; print(json.dumps(list(sys.version_info[:2])))'%dependency],
                env=selected,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,check=True)
            self.assertEqual(json.loads(result.stdout.strip().splitlines()[-1]),[3,minor])
            self.assertEqual(original['BASELINE_PYTHON_KIND'],polluted)


if __name__=='__main__':unittest.main()
