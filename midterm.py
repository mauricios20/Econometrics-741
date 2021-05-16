import numpy as np
# Chapter 4 See the following definitions of matrices A and B
A = np.array([[1, 2, 5], [6, 2, 8]])
B = np.array([[5, 0], [2, 1]])

# Find A'
AT = A.transpose()
print(f'Tranpose of A: {AT}')

# Find B'
BT = B.transpose()
print(f'Tranpose of B: {BT}')

# # Find the inverse of A
# Ainv = np.linalg.inv(A)
# print(f'Inverse of A: {Ainv}')

# Find the inverse of B
Binv = np.linalg.inv(B)
print(f'Inverse of B: {Binv}')

# # Find A+B?
# Sum = A + B
# print(f'A+B: {Sum}')

# Find A×B?
AXB = A.dot(B)
print(f'AxB: {AxB}')

# Find BXA?
BXA = B.dot(A)
print(f'BXA: {BXA}')

# Find A'B?
ATXB = AT.dot(B)
print(f'ATXB: {ATXB}')
