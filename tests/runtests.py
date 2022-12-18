import os
import runpy
import sys
from pathlib import Path


tests = [f for f in os.listdir(Path(__file__).parent)
         if f.startswith('test_') and f.endswith('.py')]


if __name__ == '__main__':
    if len(sys.argv) > 2:
        raise RuntimeError('Only no argument or a single argument that'
                           ' specifies the test file is allowed')

    if len(sys.argv) == 2:
        filename = Path(sys.argv[1]).with_suffix('.py')
        tests = [filename]

    for testfile in tests:
        path = Path(__file__).parent / testfile
        runpy.run_path(path, run_name='__main__')
