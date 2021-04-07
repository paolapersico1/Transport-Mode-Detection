import matplotlib.pyplot as plt
years=[2000, 2002, 2004, 2006, 2008, 2010]
values=[10.3, 13.23, 15.12, 16.12, 17.13, 18.67]
plt.title("Plot test")
plt.ylabel("Current values")
plt.xlabel("Years")
plt.plot(years, values, color='green', marker='o')
plt.show()

