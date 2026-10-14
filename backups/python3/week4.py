str1 = ""
if len(str1) == 0:
    print("String is empty")
if not str1:
    print("String is empty")

# convert char to int 
print(ord('a'))  # Output: 97
print(ord('A'))  # Output: 65

# convert int to char
print(chr(65))
print(chr(98))

lst = [0,0,0,0,0,0,0]
print(len(lst))

lst2 = [0]*17
print(lst2)

lst3 = [2 for _ in range(20)]
print(lst3)

lst4 = [i for i in range(2,23)]
print(lst4)
lst5 = [i+2 for i in range(21)]
print(lst5)

lst6 = [i for i in range(10,-1,-2)]
print(lst6)

# xx.sort() sorted(xx)
nums = [5,3,8,6,7,2]
nums.sort()
print(nums)
nums.sort(reverse=True)
print(nums)

sorted_nums = sorted(nums)
print(sorted_nums)
sorted_nums2 = sorted(nums, reverse=True)
print(sorted_nums2)

str1 = "talent is enduring patient"

str2 = "[i+2 for i in range(10)]"

str3 = sorted(str1)
print(str3)

# print a -> z
for x in range(97,123):
    print(chr(x), end=" ")

print()

lst1 = [1,9,9,4]
lst2 = [1,9,9,4]
print(lst1 == lst2)

dict1 = {'a':1, 'b':2}
dict2 = {'b':2,'a':1}
print(dict1 == dict2)

set1 = {1,2,3}
set2 = {3,2,1}
print(set1 == set2)

lst3 = [4,9,9,1]
print(lst1 == lst3)

set4 = set()
set5 = set()
print(set4 == set5)


str1 = "talent is enduring patient"
dic = {}
for s in str1:
    dic[s] = dic.get(s,0) + 1
print(dic)

lst1 = [1,9,9,5]
lst2 = [1,9,9,5]
print(lst1 == lst2)
lst3 = [5,9,9,1]
print(lst1 == lst3)

set1 = {1,9,9,5}
set2 = {5,9,1}
print(set1 == set2)

dic1 = {'a':1,'b':2,'c':3}
dic2 = {'c':3,'b':2,'a':1}
print(dic1 == dic2)


s = "anagram"
t = "nagaram"

freq = [0]*26

for ch in s:
    freq[ord(ch)-97] += 1
for ch in t:
    freq[ord(ch)-97] -= 1
print(freq)
for i in freq:
    if i != 0: print(False)
print(True)

s = "anagram"
t = "nagaram"

for ch in s:
    print(ch)
print(s[0])
# if char in s, freq + 1, if char in t, freq -1
dic = {}
for ch in s:
    dic[ch] = dic.get(ch,0) + 1

for ch in t:
    dic[ch] = dic.get(ch,0) - 1
print(dic)

for k,v in dic.items():
    print(k,v)
for k in dic.keys():
    print(k)
for v in dic.values():
    print(v)