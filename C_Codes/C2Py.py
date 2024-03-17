import os
import subprocess

print("Inside Python file .....")
subprocess.call(["g++", "Test.cpp"])
subprocess.call("./a.out", shell=True)



print("Task is done.")