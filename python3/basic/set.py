s1 = {2,0,1,0}

print(s1)

s1.add(10)
s1.add(1)

print(s1)

# s1.remove(20) # will throw KeyError if 20 is not present
s1.discard(0)
s1.discard(20)

print(s1)

if 1 in s1:
    print("1 is in set")
elif 2 in s1:
    print("2 is in set")
else:
    print("else")

for s in s1: print(s)

s1.clear()

empty_set = set()
