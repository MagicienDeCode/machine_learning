"""
lst = [[j+1 for j in range(i*9, (i+1)*9)] for i in range(9)]
print(lst)

[[1,  2,  3,  4,  5,  6,  7,  8,  9],
 [10, 11, 12, 13, 14, 15, 16, 17, 18], 
 [19, 20, 21, 22, 23, 24, 25, 26, 27], 
 [28, 29, 30, 31, 32, 33, 34, 35, 36], 
 [37, 38, 39, 40, 41, 42, 43, 44, 9], 
 [46, 47, 48, 49, 50, 51, 52, 9, 54], 
 [55, 56, 57, 58, 59, 60, 61, 62, 63], 
 [64, 65, 66, 67, 68, 69, 70, 71, 72], 
 [73, 74, 75, 76, 77, 78, 79, 80, 81]]

# 45 -> 9
# 53 - > 9

lst[4][-1] = 9
lst[5][-2] = 9

row_sets = [set() for _ in range(9)]
col_sets = [set() for _ in range(9)]
box_sets = [set() for _ in range(9)]

for i in range(9):
    for j in range(9):
        row_sets[i].add(lst[i][j])
        col_sets[j].add(lst[i][j])

lst = [[j+1 for j in range(i*7, (i+1)*7)] for i in range(7)]
lst[4][-1] = 29
print(lst)
for row in lst:
    print(row)


for row in lst:
    for v in row:
        print(v)


row_sets = [set() for _ in range(7)]
print(row_sets)
#row_sets[1].add(9)

for i in range(7):
    for j in range(7):
        row_sets[i].add(lst[i][j])


for i in range(7):
    if i == 3: break
    print(i)
"""

lst = [[False for _ in range(9)] for _ in range(9)]
lst2 = [[False] * 9 for _ in range(9)]
lst3 = [[False] * 9] * 9
print(lst == lst2 == lst3)
lst[-1][2] = True
lst2[-1][2] = True
lst3[-1][2] = True

lst4 = [1,9,9,4]
lst5 = [lst4] * 4

lst4[0] = 200

b1 = False 
lst6 = [b1] * 9
b1 = True
print(lst6)

lst7 = ["5","3",".",".","7",".",".",".","3"]
lst_bool = [False] * 9

for l in lst7:
    if l == ".": continue
    if lst_bool[int(l)-1]: print("Duplicate", l)
    lst_bool[int(l)-1] = True

board = [["8","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","9"]
,[".",".",".",".","8",".",".","7","9"]]

row_bool = [[False] * 9 for _ in range(9)]

for i in range(9):
    for j in range(9):
        value = board[i][j]
        if value == ".": continue
        print(value)
