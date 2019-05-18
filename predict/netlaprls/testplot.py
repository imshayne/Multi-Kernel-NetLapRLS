# _*_ coding: utf-8 _*_
__author__ = 'mcy'
__date__ = '2019-05-17 15:47'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score





x = np.arange(100)
y = np.sin(x)
fig = plt.figure()
ax = fig.add_subplot(111)

plt.plot(x, y)

plt.show()

x = np.random.randint(0,2, size=16).reshape((4, 4))
print x
y = np.random.randint(0,2, size=16).reshape((4, 4))
print y
print accuracy_score(x, y, normalize=True)