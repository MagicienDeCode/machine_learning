import pandas as pd
import sys
import random

class Japanese:
    def __init__(self, hiragana, katakana, romaji):
        self.hiragana = hiragana
        self.katakana = katakana
        self.romaji = romaji

def load_data(file_path):
    try:
        data = []
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} entries from {file_path}")
        hiragana = df['hiragana'].tolist()
        katakana = df['katakana'].tolist()
        romaji = df['romaji'].tolist()
        for i in range(len(hiragana)):
            data.append(Japanese(hiragana[i], katakana[i], romaji[i]))
        return data
    except Exception as e:
        print(f"Error reading the file: {e}")

def read_int(prompt,min_value, max_value):
    while True:
        try:
            value = int(input(prompt))
            if min_value <= value <= max_value:
                return value
            else:
                print(f"Please enter a value between {min_value} and {max_value}.")
        except ValueError:
            print("Invalid input. Please enter an integer.")
            return value
        except ValueError:
            print("Invalid input. Please enter an integer.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python game.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    data = load_data(file_path)

    option = read_int("Choose an option: 1.Hiragana 2.Katakana 3.Exit : ", 1, 3)
    
    if option == 3: sys.exit(0)

    random.shuffle(data)
    total = 1
    errors = []
    for item in data:
        input_romaji = ""
        if option == 1:
            input_romaji = input(f"No.{total} Enter the romaji for {item.hiragana}: ")
        elif option == 2:
            input_romaji = input(f"No.{total} Enter the romaji for {item.katakana}: ")
        if input_romaji.lower() == "end": break
        total += 1
        if input_romaji.lower() == item.romaji.lower():
            print(" --- Correct!")
        else:
            print(f" XXX Incorrect! The correct answer is {item.romaji}.")
            errors.append(item)

    if errors:
        print("You made some mistakes:")
        for item in errors:
            print(f"Hiragana: {item.hiragana}, Katakana: {item.katakana}, Romaji: {item.romaji}")
    
    print(f"You answered {total} questions. Your score is {total - len(errors)} / {total}. correct rates: {((total - len(errors)) / total) * 100:.2f}%")
    print("Thank you for playing!")
    sys.exit(0)