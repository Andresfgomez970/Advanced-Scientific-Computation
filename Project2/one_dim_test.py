from basic_vegas import *
import os

for i in range(100):
    os.system("python serial_vegas.py")
    os.system("python parallel_vegas.py")
