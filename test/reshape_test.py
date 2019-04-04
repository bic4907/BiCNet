import torch

a = torch.randint(low=1, high=100, size=(4, 5, 2))
'''
print(a)
# print(a.reshape(5, 2, 5))
# print(a.transpose(0, 1))
b = a.reshape(20, 2)
print(b)
print(b.reshape(4, 5, 2))
'''


