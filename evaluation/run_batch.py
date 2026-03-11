from main import main_override
import sys

def run_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for i, line in enumerate(lines, start=1):
        print(f"\n=== Running prompt {i}: {line} ===")
        main_override(line)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_batch.py testset.txt")
        sys.exit(1)
    
    run_from_file(sys.argv[1])