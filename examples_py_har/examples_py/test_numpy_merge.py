import numpy as np
a=np.array([[1,2],[3,4]])
b=np.array([[5,6],[7,8]])
#Merge on rows
c=np.concatenate((a,b))
print(c)
#Merge on columns
d=np.concatenate((a,b),axis=1)
print(d)
