"""
It is necessary to run test_numpy_like.py and test_scipy_like.py separately in a new interpreter to ensure the tests are executed correctly.
"""

import subprocess
from pathlib import Path


def main():
    this_path = Path(__file__).parent
    for file in this_path.glob("test_*.py"):
        subprocess.run(["python", "-m", "pytest", str(file)])


if __name__ == "__main__":
    main()
