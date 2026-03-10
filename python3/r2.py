# List Stack
lst = []
lst.append(1)
lst.append(9)
print(lst)
print(lst[-1])
print(lst.pop())
print(lst.pop())
print(lst)
print(len(lst) == 0)
print(not lst)
# set
s = set()
s.add(1)
s.add(9)
s.add(9)
print(s)
s.discard(100)
s.discard(1)
print(s)
# dict
d = {}
d['a'] = 1
d[3213] = "9"
print(d)
d['a'] = 100
