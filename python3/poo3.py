class Person:
    def __init__(self,name,age=18):
        self.name = name
        self.age = age
        self.height = 100

    def introduce_self(self):
        print("my name is", self.name, self.age, self.height)

p1 = Person("test",20)
p2 = Person("test2")

class Chinese(Person):
    def introduce_self(self):
        print("我的名字", self.name, self.age, self.height)


c1 = Chinese("伦")
c1.introduce_self()
p4 = Person()
p4.introduce_self()