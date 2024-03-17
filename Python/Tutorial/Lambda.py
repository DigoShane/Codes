#The Objective of this tutorial is to understand lambda and its use in python.

#============================================================================
# How does lambda compare to standard function definitions
#============================================================================
#Consider a function func that adds 5 to every no. we could code it as follows
def func1(x):
 return x+5

#We can achieve the same thing with lambda as follows
func2 = lambda x: x+5
#Note the function has to fit in 1 line

print(func2(9))
print(func1(9))

#============================================================================
# Defining function in another function.
#============================================================================

def func3(x):
 func4 = lambda x: x+5
 return func4(x)+10

print(func3(9))


#============================================================================
# Defining lambda with multiple variables
#============================================================================

func5 = lambda x,y: x+y

print(func5(5,5))


#============================================================================
# Defining lambda with optional variables
#============================================================================

func6 = lambda x,y=4: x+y

print(func6(5))

#============================================================================
# Using lambda with the map function
#============================================================================
#defining a map function where the function is defined using lambda
a = [1,2,3,4,5,6,7,8,9,10]
newList1 = list(map(lambda x: x+5,a))

print(newList1)

#============================================================================
# Using lambda with the filter function
#============================================================================
#defining a filter function where the function is defined using lambda
#A filter function filters out elements based on the condition in its first argument.
a = [1,2,3,4,5,6,7,8,9,10]
newList2 = list(filter(lambda x: x%2==0,a))

print(newList2)








