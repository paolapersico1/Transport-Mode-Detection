import matplotlib.pyplot as plt
import numpy as np
import math

label=["2000", "2002", "2004", "2006", "2008", "2010"]
valuesX=[0,1,2,3,4,5]
valuesY=[10.3, 13.23, 15.12, 16.12, 17.13, 18.67]
plt.axis([-1,6,10,19])
plt.bar(valuesX, valuesY, 0.5)
plt.xticks(np.arange(6), label)
plt.ylabel("Current values")
plt.xlabel("Years")
plt.show()

