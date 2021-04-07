

def main():
	print("Simple Python list example (version 2: use lambda function)")
	square=lambda x:x**2
	mylist=[]
	i=0
	while (i<=10):
		mylist.append(square(i))
		i+=1
	print(mylist)		

main()
