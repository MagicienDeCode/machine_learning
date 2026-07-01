from pathlib import Path
import shutil

current = Path(".")

# pwd
print(current.cwd())
print(current.exists())
print(current.is_file())
print(current.is_dir())

# list all files in the current directory: ls
for file in current.iterdir():
    print(file)

# list only files or only folders

print("="*10)

# list only py files
for file in current.iterdir():
    if file.suffix == ".py":
        print(file)

print("="*20)

# cd then mkdir t1
Path("./t1").mkdir(parents=True, exist_ok=True)

# cp lst.py to test.txt
shutil.copy("lst.py", "test.txt")

# cp src_dir to dst_dir, overwrite if dst_dir exists
shutil.copytree("t1", "test", dirs_exist_ok=True)

# mv test.txt to test222.txt
shutil.move("test.txt", "test222.txt")
Path("test222.txt").rename("test333.txt")

# rm test333.txt
toDelete = Path("test333.txt")
if toDelete.exists():
    toDelete.unlink()

# rm empty folder t1
Path("t1").rmdir()
# rm non-empty folder test, rm -R 
#shutil.rmtree("to be careful")

# read file
with open("test.py", "r", encoding="utf-8") as f:
    text = f.read()

print(text)