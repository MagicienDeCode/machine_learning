s = "anagram"
t = "nagaram"

dic = {}
for ch in s:
    dic[ch] = dic.get(ch,0) + 1
for ch in t:
    dic[ch] = dic.get(ch,0) - 1

for v in dic.values():
    if v != 0:
        print(False)


dic.clear()
print(s==t)
print(s[0])
print(t[0])

# create a dictionary to store index of characters
for i in range(len(s)):
    if s[i] in dic:
        dic[s[i]].append(i)
    else:
        dic[s[i]] = [i]

print(dic)
""" !! not correct !!
dic2 = {}
for i in range(len(t)):
    dic2.get(t[i], []).append(i)
"""

from collections import defaultdict
dic2 = defaultdict(list)

for i in range(len(t)):
    dic2[t[i]].append(i)
print(dic2)

dic3 = {}
for i in range(len(t)):
    dic3.setdefault(t[i], []).append(i)

strs = ["eat","tea","tan","ate","nat","bat"]


sorted_s = ["".join(sorted(s)) for s in strs]
print(sorted_s)

dic4 = {}
"""
s = "eat"
key = "".join(sorted(s))  # "aet"
"""

for s in strs:
    key = "".join(sorted(s))
    dic4.setdefault(key, []).append(s)

print(dic4)


test = "aA"
test2 = "".join(sorted(test))


dic2 = {}
dic2['abt'] = ['bat']
print(dic2)
dic2.clear()

from collections import defaultdict
dic3 = defaultdict(list)
for s in strs:
    key = "".join(sorted(s))
    dic3[key].append(s)
print(dic3)


