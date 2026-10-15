lst = [2,0,1,0]

print(lst)

print(lst[0])
print(lst[1])
print(lst[-1])
print(lst[len(lst)-1])

lst.append(10)
print(lst)

print(lst.pop())
print(lst)

lst.sort()
print(lst)

new_lst = sorted(lst,reverse=True)
print(new_lst)

for i in range(len(lst)):
  print(i, lst[i])
for l in lst: print(l)
for i,v in enumerate(lst): print(i,v)

slst = ["python3","kotlin","java"]
# write a function count how many elements in a string list have length >= 5, return the count


