"""
test_numpy_like.py and test_scipy_like.py must be run seperately
"""

import os
import subprocess
import sys
from pathlib import Path


def main(file_names):
    this_path = Path(__file__).parent.resolve()

    if not file_names:
        files = this_path.glob("test_*.py")
    else:
        files = [this_path / name for name in file_names]

    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{str(this_path)}{os.pathsep}{existing_pp}" if existing_pp else str(this_path)

    failed = False
    for file in map(str, files):
        res = subprocess.run(
            [sys.executable, "-m", "pytest", file, "--tb=native", "-s"],
            cwd=str(this_path),
            env=env,
        )
        if res.returncode != 0:
            failed = True

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
