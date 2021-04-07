
def square(x):
	return x**2 



def main():
	print("Simple Python list example (version 1)")
	mylist=[]
	i=0
	while (i<=10):
		mylist.append(square(i))
		i+=1
	print(mylist)		

main()
