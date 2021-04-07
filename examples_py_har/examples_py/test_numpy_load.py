import numpy as np
myfile="iris.csv"

#Load columns 0,1,2,3 and remove the header (row 1)
data=np.loadtxt(myfile, delimiter=',', usecols={0,1,2,3}, skiprows=1)

#Perform some slicing (results will be (150,4))
print(data[:,:].shape)

#What will be the output here?
print(data[5:,:].shape)

#What will be the output here?
print(data[5:,1:2].shape)

#What will be the output here?
print(data[5:10,1:2].shape)



