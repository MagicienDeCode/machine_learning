from typing import List


# basic functions

print("ni hao") #

# convert str to int
print(int("9")) # 

# max of 3 elements
print(max(7,8,9)) # 

# convert int to str
print(str(1)) # 

# retrieve length of a list, 
# a set, a dictionary, a string
print(len("list set dic ...")) # 

# verify if a string is empty
print(not "") # 

# process control
if 3 > 2 :
	print("3>2") # 
elif 4 > 3 :
	print("4>3") #
else:
	print("else") #

counter = 10
while counter > 7:
	print(counter) # 
	counter -= 1

name = "xiang"
for i in range(len(name)):
	print(name[i],ord(name[i])) # 
for i in range(4):
	print(i) # 
for i in range(0,4,2):
	print(i) # 
for i in range(10,7,-1):
	print(i) # 
for i in range(10,7,-1):
	if i == 10:
		break
	print(i) # 
for i in range(10,7,-1):
	if i == 10:
		continue
	print(i) # 


# list

my_list = [1,9,9,4]

# get the first 2 elements
print(my_list[:2]) # 

# get element from index 1 to 3 inclusive
print(my_list[1:len(my_list)]) # 

# index starts from 0
print(my_list[3]) # 

# if index out of range
# print(my_list[5]) # IndexError: list index out of range

# add a value in list
my_list.append(0)
my_list.append(1)
print(my_list) # 

# combine two lists
my_list2 = [2,7]
my_list.extend(my_list2)
print(my_list) # 

# insert with index
my_list.insert(1,10)
print(my_list) # 

# iterate
for i in range(len(my_list)):
	print(my_list[i]) # 

for i in my_list:
	print(i) # 

for i,v in enumerate(my_list):
	print(i,v) # 



# dictionary

dic = {"name":"njk","age":30}
print(dic) # 

# get value
print(dic["age"]) # 

# change value
dic["age"] = 31
print(dic["age"]) # 

# add a new key,value pair
dic["pays"] = "JP"
print(dic) # 

# delete a key,value pair
del dic["name"]
print(dic) # 

# get keys or values
print(dic.keys()) # 
print(dic.values()) # 

# iterate 
for k,v in dic.items():
	print(k,v) # 

# verify if key exists
if "age" in dic:
	print(dic["age"]) # 

# clear
dic.clear()
print(dic) # 

# get value, if not exists, return default
print(dic.get("test","test no in dic")) # 
print(dic.get("test",99)) # 



# set

my_set = {1,9,9,4}
print(my_set) # 

# add a value in set 
my_set.add(2)
print(my_set) # 

# remove an element in list
# remove: if key not exists in set, KeyError, discard: if not exists, do nothing
my_set.discard(10)
# my_set.remove(10) # KeyError: 10

my_set.remove(9)
print(my_set) # 

# create an empty set
empty_set = set()
print(empty_set) #



# stirng

str1 = "xiang"
print(str1) # 

# iterate
for i in str1:
	print(i) # 
for i in range(len(str1)):
	print(str1[i]) # 

# ord, convert char to int
print(str1[2]) # 
print(ord(str1[2])) # 
print(ord('a')) # 



# end of for: return