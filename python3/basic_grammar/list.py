from typing import List
# declare an int list with values 1,9,9,5

# print entire my_list

# print the first element in my_list 

# print the second element in my_list

# for loop, print each index of my_list

# for loop, print each element of my_list

# range

# declare a string list named str_list contains "python3","kotlin","java"

# print length of this list

# print element "java" in this list

# print each element in list using its index


# define a function named countLen5 that accept a string list as parameter and return an int as result
	# declare a variable named res, assign value 0
	# iterate list directly, when element's len >= 5, add 1 for res
	# return res



# print result: call previous function using str_list above

# def count_len5(str_lst: List[str]) -> int:

lst = ["python3","kotlin","java"]

lst2 = ["python3","kotlin","java","222","111"]

lst3 = [123231,3232,222]

# def count_len5(str_lst: List[str]) -> int:
def count_len5(str_lst): 
    result = 0
    for s in str_lst:
        if len(s) >= 5:
            result += 1
    return result

def find_and_print_len5(str_lst):
    for s in str_lst:
        if len(s) >= 5:
            print(s)
    
print(count_len5(lst))
print(count_len5(lst2))