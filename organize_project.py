import os
import shutil

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def move_file(src, dst):
    if os.path.exists(dst):
        print(f"[SKIP] {dst} already exists. Skipping {src}.")
        return
    shutil.move(src, dst)
    print(f"Moved {src} -> {dst}")

def organize():
    root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root, 'data')
    logs_dir = os.path.join(root, 'logs')
    tests_dir = os.path.join(root, 'tests')
    ensure_dir(data_dir)
    ensure_dir(logs_dir)
    ensure_dir(tests_dir)

    # Move CSV and JSONL files to data/
    for fname in os.listdir(root):
        if fname.endswith('.csv') or fname.endswith('.jsonl'):
            move_file(os.path.join(root, fname), os.path.join(data_dir, fname))

    # Move log files to logs/
    for fname in os.listdir(root):
        if fname.endswith('.log') or (fname.endswith('.csv') and 'log' in fname):
            move_file(os.path.join(root, fname), os.path.join(logs_dir, fname))

    # Move test scripts to tests/
    for fname in os.listdir(root):
        if fname.startswith('test_') and fname.endswith('.py'):
            move_file(os.path.join(root, fname), os.path.join(tests_dir, fname))

    # Add .gitignore if not present
    gitignore_path = os.path.join(root, '.gitignore')
    if not os.path.exists(gitignore_path):
        with open(gitignore_path, 'w') as f:
            f.write("""
.env
jobs.db
logs/
__pycache__/
*.pyc
""")
        print("Created .gitignore with recommended patterns.")
    else:
        print(".gitignore already exists.")

    print("\nOrganization complete!")

if __name__ == "__main__":
    organize()