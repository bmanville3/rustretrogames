import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

PATTERN = re.compile(
    r"Episode (\d+): average reward over last 100 episodes = ([\-\d.]+) with epsilon = ([\d.]+)"
)

def existing_file(path_string: str) -> Path:
    path = Path(path_string).absolute()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"{path} does not exist. Please pass in a file that exists")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"{path} must be a file")
    return path

parser = argparse.ArgumentParser()
parser.add_argument("file", type=existing_file, help="Path to an existing log file.")

def log_file_input(log_file: str | Path) -> None:

    episodes = []
    rewards = []
    epsilons = []

    log_file = Path(log_file)
    with open(log_file, "r") as f:
        for line in f:
            match = PATTERN.search(line)
            if match:
                episodes.append(int(match.group(1)))
                rewards.append(float(match.group(2)))
                epsilons.append(float(match.group(3)))

    plt.figure(figsize=(12,6))
    plt.scatter(episodes, rewards, c='blue', label='Average Reward')

    for x, y, eps in zip(episodes, rewards, epsilons):
        plt.text(x, y, f"{eps:.2f}", fontsize=8, ha='right', va='bottom')

    coefficients = np.polyfit(episodes, rewards, deg=1)
    poly = np.poly1d(coefficients)
    plt.plot(episodes, poly(episodes), color='red', linewidth=1.5, alpha=0.5, label='Linear best-fit line')

    plt.plot(episodes, rewards, linestyle='--', color='gray', alpha=0.5)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward (last 100 episodes)")
    plt.title(f"Reward vs Episodes from log: {log_file.name}")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    args = parser.parse_args()
    log_file_input(args.file)
