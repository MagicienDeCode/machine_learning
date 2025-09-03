import os
import pandas as pd
import platform
import subprocess
import time
from wcwidth import wcswidth
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
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the japanese directory path (two levels up)
#japanese_dir = os.path.dirname(current_dir)
japanese_dir = current_dir

def pad_display(s, total_width):
    return s + ' ' * (total_width - wcswidth(s))

def read_int(prompt, min_value, max_value):
    while True:
        try:
            value = int(input(prompt))
            if min_value <= value <= max_value:
                return value
            else:
                print(f"Please enter a number between {min_value} and {max_value}.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

if __name__ == "__main__":
    display_options = ""
    for i,f in enumerate(files):
        display_options += f"{i+1}. {f}\n"
    options = read_int(f"Select a file to play: \n{display_options}: ", 1, len(files))
    
    selected_file = files[options - 1]
    play_dir = os.path.join(japanese_dir, selected_file)

    # Check if file exists before processing
    if not os.path.exists(play_dir):
        print(f"Error: File not found - {play_dir}")
        exit(1)

    print(f"Playing files from {play_dir}...")
    df = pd.read_csv(play_dir)
    first_column = df.columns[0]

    dir_path = selected_file.split('.')[0]
    save_dir = os.path.join(current_dir, "downloads" ,dir_path)
    total_rows = len(df)

    skip = read_int(f"Enter the number of rows to skip (0 for no skip) max {total_rows}: ", 0, total_rows)

    for index, row in df.iterrows():
        if index < skip:
            continue
        first_col_value = row.iloc[0]  # 取第一列值
        word_display = pad_display(f"{index+1}/{total_rows} : ", 10)
        for r in row:
            if str(r) != "nan":
                word_display += pad_display(r, 15)
        print(word_display)
        audio_path = os.path.join(save_dir, f"{first_col_value}.mp3")
        if not os.path.exists(audio_path):
            print(f"Error: Audio file not found for {first_col_value}.")
            continue
        system = platform.system()
        if system == "Windows":
            ffplay_cmd = "ffplay.exe"
        else:
            ffplay_cmd = "ffplay"
        subprocess.run(
            [ffplay_cmd, "-nodisp", "-autoexit", "-loglevel", "quiet", audio_path]
        )
        time.sleep(0.5)

print("=======> Finished playing all files.")
