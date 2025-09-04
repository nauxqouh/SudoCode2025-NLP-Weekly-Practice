import os
import subprocess

def load_viwik18(output_path="data/raw/viwik18.txt"):
    # Clone repo
    if not os.path.exists("viwik18"):
        subprocess.run(["git", "clone", "https://github.com/NTT123/viwik18"])
    
    # Merge file
    with open(output_path, "w", encoding="utf-8") as outfile:
        for part in sorted(os.listdir("viwik18/dataset")):
            part_path = os.path.join("viwik18/dataset", part)
            with open(part_path, "r", encoding="utf-8") as infile:
                outfile.write(infile.read())

    print(f"Dataset saved to {output_path}")
    return output_path