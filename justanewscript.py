import numpy as np
import torch
import matplotlib.pyplot as plt 

for i in range(100):
    base = float(input('base: \n'))
    rate = float(input('rate: \n'))
    var = float(input('var: \n'))
    if var > 0.15:
        var = 0.15
    time = int(input('time: \n'))
    inc = float(input('inc: \n'))
    tests = []
    for i in range(10000):
        input = base
        for j in range(time):
            input = input * np.random.normal(loc=rate, scale=var)
            input = input + inc
        tests.append(input)
    tests = np.array(tests)
    print(int(np.mean(tests)))
    print(int(np.max(tests)))
    print(int(np.min(tests)))

'''
s = [2,3,5,7,11]
num = [i for i in range(s[-1] + 1,150)]
for i in s:
    for j in range(len(num)):
        if num[j] % i == 0:
            num[j] = 0
for j in num:
    if j != 0:
        s.append(j)
num = [i for i in range(s[-1] + 1, 10000)]
for i in s:
    for j in range(len(num)):
        if num[j] % i == 0:
            num[j] = 0
for j in num:
    if j != 0:
        s.append(j)
'''