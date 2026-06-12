import pandas as pd
from pathlib import Path

def main():
    csv_dir = Path("subjectdata")
    subjects = ["0004", "0006", "0007", "0008", "0009", "0010"]
    
    all_show_T = []
    
    for sub in subjects:
        csv_path = csv_dir / f"experiment_data_subject{sub}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        all_show_T.extend(df["show_T"].tolist())
        
    all_show_T = pd.Series(all_show_T)
    print("show_T statistics:")
    print(all_show_T.describe())
    print("\nFraction of show_T in different intervals (modulo 1.0):")
    mod_show_T = all_show_T % 1.0
    print(f"  0.0 to 0.25 (Falling from right / Rising from left): {((mod_show_T >= 0.0) & (mod_show_T < 0.25)).mean()*100:.2f}%")
    print(f"  0.25 to 0.50 (Rising to left / Falling to left): {((mod_show_T >= 0.25) & (mod_show_T < 0.5)).mean()*100:.2f}%")
    print(f"  0.50 to 0.75 (Falling from left / Rising from right): {((mod_show_T >= 0.5) & (mod_show_T < 0.75)).mean()*100:.2f}%")
    print(f"  0.75 to 1.00 (Rising to right / Falling to right): {((mod_show_T >= 0.75) & (mod_show_T < 1.0)).mean()*100:.2f}%")

if __name__ == "__main__":
    main()
