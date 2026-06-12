import os

def main():
    root = "d:/intuitive physics/pendulum_dataset"
    print("Searching for folders containing 'K7':")
    for r, dirs, files in os.walk(root):
        for d in dirs:
            if "K7" in d:
                print(f"Found dir: {os.path.join(r, d)}")
        for f in files:
            if "K7" in f:
                print(f"Found file: {os.path.join(r, f)}")

if __name__ == "__main__":
    main()
