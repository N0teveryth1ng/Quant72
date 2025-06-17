# from sklearn.ensemble import VotingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from src.preprocess import X ,y
#
# model1 = LogisticRegression()
# model2 = DecisionTreeClassifier()
#
# ensemble = VotingClassifier(
#    estimators=[('lr', model1), ('dct', model2)],
#     voting='sot'
#   )
#
# ensemble.fit(X , y)
# from tabnanny import check

from fontTools.misc.cython import returns
from nltk.misc.chomsky import subjects
from nltk.sem.chat80 import country
from sympy.printing.tree import print_node

# cubes = {x: x**3 for x in range(1, 11) if x % 2 != 0}
# print(cubes)
#
#
# for x in range(1,11):
#     if x % 2 != 0: print(x)


# import numpy as np
# matrix1 = np.array([[1, 2], [3, 4]])
# matrix2 = np.array([[4, 5], [5, 7]])
# print("Mat-mul:\n", np.linalg.matmul(matrix1,matrix2))  # No loops!


# s = input('enter: ')
# if s == s[::-1]:
#   print('palindrome')
# else:
#   print('none')



# saarc = [ "Afghanistan", "Bangladesh", "Bhutan", "India", "Maldives", "Nepal", "Pakistan", "Sri Lanka" ]
# enter = input()
# print("Member" if enter in saarc else "No member")



# simple interest calculator
# def calculator(p=1000, t=5, r=10):
#     return (p * t * r) / 100
# print(calculator(2000,3, 16))


# leap year
# input = int(input('enter: '))
# if (input % 4 == 0 and input % 100 != 0) or input % 4 == 0:
#    print('Leap')
# else:
#    print('None')


# child or not
# age = int(input('enter: '))
# if age <= 13:
#     print('child')
# elif age < 20:
#      print('adult')
# else:
#     print('Senior')


# seasons = ['Spring', 'Summer', 'Fall', 'Winter']
# for x,seasons in enumerate(seasons):
#     print(f"index: {x} seasons: {seasons}")



# non odd prime nums
# for i in range(1,51):
#     if i % 2 != 0:
#         for j in range(2, i):
#             if i % j == 0:
#                 print(i)
#                 break

# sum of digits
# n = int(input("enter: "))
# s = 0
# while n > 0:
#     s += n % 10
#     n //= 10
# print(s)


# pattern print
# n = 4
# for i in range(1, n+1):
#     print(str(i)*i)

# sum of series
# n = int(input())
# s = 0
# for i in range(1, n+1):
#   s += i/(i + 1)
# print(s)

# for i in range(5):
#     if i == 3:
#         break
#     print("l1",i)
#
# for i in range(5):
#     if i == 3:
#         continue
#     print("l2",i)

# simple inetrets
# def simple_int(p=1000, t=5, r=10):
#     return (p * t * r) / 100
#
# print(simple_int())
# print(simple_int(2000))
# print(simple_int(2000, 2,5))

# factorial
# def fact(n):
#     f = 1
#     for i in range(1, n+1):
#         f *= i
#     return f
#
# print(fact(int(input("enter: "))))
#
# def sports(name, *sports):
#     print("name: ",name)
#     print("sports: ", sports)
#
# sports("soham", "cricket", "tennis")


# s = lambda x: "even" if x % 2 == 0 else "odd"
# print(s(int(input("enter: "))))

# fibbonaci using reccursion:

# def fib(n):
#     if n <= 1: return n
#     return fib(n-1) + fib(n-2)
#
# for i in range(10):
#     print(fib(i), end=' ')
#

#
# class account:
#     count = 0
#     def __init__(self, name, acc_no, balance):
#         self.name = name
#         self.acc_no = acc_no
#         self.balance = balance
#         account.count += 1
#
#     def deposit(self, amt):
#         self.balance += amt
#
#     def withdraw(self, amt):
#         if amt <= self.balance:
#             self.balance -= amt
#
#     def display(self):
#         print("Name -", self.name, "| Acc No -",self.acc_no, "| Bal -",self.balance)
#
# # using the bank
# a1 = account("Amit", 1234, 5000)
# a1.withdraw(3000)
#
# a1.display()
#
# #  student class
# class student:
#     def __init__(self, name, role, marks):
#         self.name = name
#         self.role = role
#         self.marks = marks
#
#     def display(self):
#         print(self.name, self.role, self.marks)
#
#
# s1 = student("Wrick", 42, 45)
# s1.display()

# 28 -
class person:
    def __init__(self, name, age, ):
        self.name = name
        self.age = age

    def display_person(self):
        print("Name -",self.name, "| Age -",self.age)


class student(person):
  def __init__(self, name, age, roll, marks):
    super().__init__(name, age)
    self.roll = roll
    self.marks = marks

  def display_student(self):
        self.display_person()
        print("roll -", self.roll, "| marks -", self.marks)

class teacher(person):
    def __init__(self, name, age, subject, experience):
        super().__init__(name, age)
        self.subject = subject
        self.experience = experience

    def display_teacher(self):
        self.display_person()
        print("subject -", self.subject, "| exp -", self.experience)

    """  using the classes   """
s = teacher("alec", "19", "english", 6)
a = student("wrick", 12, 78, 89)
s.display_teacher()
a.display_student()


# method overloading
class animal:
    def speak(self):
        print("animal speaks")

class dog(animal):
    print("dog barks")

d = dog()
d.speak()

# palindrome check
s = input("enter -")
if s == s[::-1]:
    print("palindrome")
else:
    print("not palindrome")


# zip
Q = ['cap of india?']
A = ["delhi"]
for Q,A in zip(Q,A):
    print(f"Q: {Q}: A: {A}")