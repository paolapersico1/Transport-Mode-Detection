import numpy as np

#Create the arrays
A=np.array([[1,2,3],[4,5,6]],dtype=np.int16)
B=np.array([[1,1,1],[0,0,0]])

# Multiply the arrays
C=A*B
print("Array result C=A*B")
print(C)
print("Shape: "+str(C.shape)+" Dim: "+str(C.ndim)+" Size: "+str(C.size));

#Reshape the array
D=C.reshape(3,2)
print(D)

#Modify the array
D[0,0]=47

#Print the original array
#See the element in position (0,0)
print(C)
