"""
Run all tests separately, because test_numpy_like.py and 
test_scipy_like.py cannot be run together with the other tests.
"""

import subprocess
from pathlib import Path


def main():
    this_path = Path(__file__).parent
    for file in this_path.glob("test_*.py"):
        subprocess.run(["python", "-m", "pytest", str(file)])


if __name__ == "__main__":
    main()
