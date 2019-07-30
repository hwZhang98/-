import numpy as np
a = np.array([1,2,3])
b = np.array([[2,3],[4,5]])
C = a>1
a[a>1] = 5
print(a)
print(np.where(C==True))
b[a>1][1] = 0
print(b)