dic = {"name": "hanzawa naoki", "age": 30, "city": "kanazawa"}

print(dic)
print(dic["name"])

dic["country"] = "France"
print(dic)
dic["age"] = 33
del dic["city"]
print(dic)

if "city" in dic:
    print("city in dic")
elif "name" in dic:
    print("name in dic")
else:
    print("else")

print(dic.keys())
print(dic.values())

for k,v in dic.items():
    print(k,v)

print(dic.get("name","if not found"))
print(dic.get("test","if not found"))


 