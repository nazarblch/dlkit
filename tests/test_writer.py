import os
import sys
import subprocess
import tempfile
import unittest


CONFIG = {
    'STRING': 'string',
    'TUPLE': ('tuple',),
    'INT': 123,
    'FLOAT': 1e-3,
}

DOC = 'docstring'

SCRIPT = """
'''{doc}'''
from dlkit.writer import Writer

{config}

writer = Writer('{root}')
"""


class TestWriter(unittest.TestCase):

    def test_writer(self):
        with tempfile.TemporaryDirectory() as workspace:
            root = os.path.join(workspace, 'test')
            config = '\n'.join(f'{k} = {repr(v)}' for k, v in CONFIG.items())
            script = os.path.join(workspace, 'script.py')
            code = SCRIPT.format(doc=DOC, config=config, root=root)
            with open(script, 'w') as f:
                f.write(code)
            process = subprocess.Popen([sys.executable, script])
            process.wait(30)
            self.assertEqual(process.returncode, 0)


if __name__ == '__main__':
    unittest.main()
