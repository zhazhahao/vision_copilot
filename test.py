import multiprocessing
import signal
import os
import time

class Parent1:
    def __iter__(self):
        print("Iterating from Parent1")
        return iter([1, 2, 3])

class Parent2:
    def __iter__(self):
        print("Iterating from Parent2")
        return iter([4, 5, 6])

class Child(Parent1, Parent2):
    def __iter__(self):
        print("Iterating from Child")
        # Calling Parent1's __iter__() explicitly
        return Parent1.__iter__(self)

if __name__ == "__main__":
    child = Child()
    for item in child:
        print(item)
