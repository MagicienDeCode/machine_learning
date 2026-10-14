class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def chage_name(self,new_name):
        self.name = new_name

    def test():
        print("test")

p1 = Person("test",20)
print(p1.age, p1.name)

p1.chage_name("france")

print(p1.age, p1.name)

p1.test()

class Solution:
    def twoSum(self):
        print("twoSum")

solution = Solution()


"expected result" 
# [0,1] ok
# [1,0] ok
{[0,1], [1,0]}
if  solution.twoSum() in {[0,1], [1,0]}:
    print("ok")

[0,1] == solution.twoSum()

