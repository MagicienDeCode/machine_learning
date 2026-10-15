lst = [
[1, 2, 3],
[1, 2, 3],
[1, 2, 3],
[1, 2, 3],
[1, 2, 3]
]
print(lst) #


lst2 = [[1,2,3] for _ in range(5)]
lst3 = [[1,2,3]*5]
print(lst2) 
print(lst3)


blst = [False, False, False]
blst[1] = True
print(blst)


b3 = [
[False, False, False],
[False, False, False],
[False, False, False],
[False, False, False],
[False, False, False]
]

print(b3)

b4 = [[False]*3 for _ in range(5)]
print(b4==b3)
b5 = [[False]*3]*5
print(b5==b3)

b3[-1][1] = True
print(b3)

b4[-1][1] = True
b5[-1][1] = True
print(b3 == b4 == b5)