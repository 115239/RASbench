import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from paper200.model_service import source_binding


class ImportedSourceTests(unittest.TestCase):
    def test_matching_contents_in_wrong_runtime_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);actual=d/'mutable.py';expected=d/'snapshot.py'
            actual.write_text('value = 1\n');expected.write_text('value = 1\n')
            module=SimpleNamespace(__file__=str(actual),__name__='router')
            with self.assertRaisesRegex(ValueError,'MODEL_SOURCE_SNAPSHOT_MISMATCH'):
                source_binding(module,expected)

    def test_imported_snapshot_identity_records_actual_sha(self):
        with tempfile.TemporaryDirectory() as d:
            expected=Path(d)/'snapshot.py';expected.write_text('value = 1\n')
            module=SimpleNamespace(__file__=str(expected),__name__='router')
            row=source_binding(module,expected)
            self.assertTrue(row['snapshot_path_matched']);self.assertEqual(len(row['sha256']),64)
            self.assertEqual(row['actual_file'],str(expected))


if __name__=='__main__':unittest.main()
