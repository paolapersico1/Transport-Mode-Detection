import matplotlib.pyplot as plt
import numpy as np
import math

X=np.arange(1,20)
serie1=[y for y in X]
serie2=[math.log(y,2) for y in X]
plt.title("Plot Multi-curves")
plt.ylabel("Y values")
plt.xlabel("X values")
plt.plot(X, serie1,'g-', label="serie1")
plt.plot(X, serie2,'r:', label="serie2")
plt.legend()
plt.show()

