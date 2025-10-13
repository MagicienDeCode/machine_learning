import os
import pandas as pd
import json

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        json_date = json.load(f)
    print("Loaded JSON:", json_date)
else:
    print(f"File {config_path} does not exist.")
    exit(-1)

files = json_date.get("files", [])
files.sort()
current_dir = os.path.dirname(os.path.abspath(__file__))
japanese_dir = current_dir
seen = set()

for file in files:
    file_path = os.path.join(japanese_dir ,file)
    if not os.path.exists(file_path):
        print(f"Error: File not found - {file_path}")
        continue

    df = pd.read_csv(file_path)
    
    duplicates = []
    for i, row in df.iterrows():
        word = row[df.columns[0]]
        if word in seen:
            duplicates.append((i, word))
        else:
            seen.add(word)
    for index, word in duplicates:
        print(f"Duplicate found in {file} at row {index + 2}: {word}")  # +2 for header and 0-based index

print("=======> Duplicate check completed.")