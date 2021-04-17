import matplotlib.pyplot as plt
import numpy as np

def plot_y(y):
    plt.bar(x=np.unique(y, return_counts=True)[0], height=np.unique(y, return_counts=True)[1])
    plt.show()
