import math
import numpy as np
import random
import sys
def fun(x):
    return x**2
for i in range(10000):
    r = random.uniform(-10.0,10.0)
    print(r,fun(r))
