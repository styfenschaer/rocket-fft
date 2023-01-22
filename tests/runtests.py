import subprocess
from pathlib import Path


def main():
    this_path = Path(__file__).parent
    for file in this_path.glob("test_*.py"):
        subprocess.run(["python", "-m", "pytest", str(file)])


if __name__ == "__main__":
    main()
