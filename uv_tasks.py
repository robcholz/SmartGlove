import subprocess
import sys


COMMANDS = [
    ["cargo", "fmt", "--all"],
    ["cargo", "clippy", "--all-targets"],
    ["cargo", "build"],
    ["cargo", "test"],
]


def verify() -> int:
    for command in COMMANDS:
        print(f"+ {' '.join(command)}", flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(verify())
