# -*- coding: utf-8 -*-
"""
Created on Wed May 10 18:00:55 2017

@author: Dipanjan
"""


## vectors

x = [1, 2, 3, 4, 5]
x

# using numpy
import numpy as np
x = np.array([1, 2, 3, 4, 5])

print(x)
print(type(x))



## matrices

m = np.array([[1, 5, 2],
              [4, 7, 4],
              [2, 0, 9]])

# view matrix
print(m)

# view dimensions
print(m.shape)


# matrix transpose
print('Matrix Transpose:\n', m.transpose(), '\n')

# matrix determinant
print ('Matrix Determinant:', np.linalg.det(m), '\n')

# matrix inverse
m_inv = np.linalg.inv(m)
print ('Matrix inverse:\n', m_inv, '\n')

# identity matrix (result of matrix x matrix_inverse)
iden_m =  np.dot(m, m_inv)
iden_m = np.round(np.abs(iden_m), 0)
print ('Product of matrix and its inverse:\n', iden_m)


# eigendecomposition
m = np.array([[1, 5, 2],
              [4, 7, 4],
              [2, 0, 9]])

eigen_vals, eigen_vecs = np.linalg.eig(m)

print('Eigen Values:', eigen_vals, '\n')
print('Eigen Vectors:\n', eigen_vecs)


# SVD
m = np.array([[1, 5, 2],
              [4, 7, 4],
              [2, 0, 9]])

U, S, VT = np.linalg.svd(m)

print ('Getting SVD outputs:-\n')
print('U:\n', U, '\n')
print('S:\n', S, '\n')
print('VT:\n', VT, '\n')





# descriptive statistics
import scipy as sp
import numpy as np

# get data
nums = np.random.randint(1,20, size=(1,15))[0]
print('Data: ', nums)

# get descriptive stats
print ('Mean:', sp.mean(nums))
print ('Median:', sp.median(nums))
print ('Mode:', sp.stats.mode(nums))
print ('Standard Deviation:', sp.std(nums))
print ('Variance:', sp.var(nums))
print ('Skew:', sp.stats.skew(nums))
print ('Kurtosis:', sp.stats.kurtosis(nums))

