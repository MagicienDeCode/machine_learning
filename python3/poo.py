b1 = False

def f(b):
    b = not b
    return b
"""
result = f(False)
print(result)
r2 = f(b1)
"""

print(f(b1))

lst = [False]

def f2(lb):
    lb[0] = not lb[0]
    return lb[0]

print(f2(lst))


s1 = "hello world"

def f3(s):
    print(s)
    s = "python"
    return s

print(f3(s1))


set1 = {1,9,9,4}

def f4(se):
    print(se)
    se.clear()

f4(set1)

print(set1)