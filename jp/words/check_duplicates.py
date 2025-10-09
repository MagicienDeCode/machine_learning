import os
import pandas as pd

files = ['hiragana.csv', 'katakana.csv']
current_dir = os.path.dirname(os.path.abspath(__file__))
japanese_dir = current_dir

for file in files:
    file_path = os.path.join(japanese_dir ,file)
    if not os.path.exists(file_path):
        print(f"Error: File not found - {file_path}")
        continue

    df = pd.read_csv(file_path)
    seen = set()
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