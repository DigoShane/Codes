#This is an exmple explaining the use of __main__.
#The example is taken from "https://www.youtube.com/watch?v=NB5LGzmSiCs&ab_channel=PythonSimplified"
#the way this is laid out is in sections. To run the code uncomment each section and run.
#A given section is a modification of the previous one.


##---------------------------------------------------------------------------------------------------
## Section 1
##---------------------------------------------------------------------------------------------------
##We first consider a top level code.
#print("this is a module")
##This is an example of a top level code


##---------------------------------------------------------------------------------------------------
## Section 2
##---------------------------------------------------------------------------------------------------
##Now consider the case when we have to import a file.
#
#import pandas #Still considered part of top level code
#
#df = pandas.DataFrame([1,2]) #This is not top level code since we have borrowed a class from a module which we have not specified in this program. It is external to this code.
#print("this is a module") #still part of top level code
#
##also code written directly in the console is also considered as top level.

##---------------------------------------------------------------------------------------------------
## Section 3
##---------------------------------------------------------------------------------------------------
##We now come to __main__ and how it helps us identify the name of the top level environment where a top level code is run.
#
#import pandas 
#
#df = pandas.DataFrame([1,2]) 
#print("this is a module") 
#print(__name__)
##if __name__==__main__ checks if we are top level or not.
#print(pandas.__name__)
##we are no longer dealing with a global variable.


##---------------------------------------------------------------------------------------------------
## Section 4
##---------------------------------------------------------------------------------------------------
##What happens if our import is very specific. From datetime, we want to import timezone
#
#from datetime import timezone 
#
#print(timezone.__name__)
#print(datatime.__name__) #Thisnwill give an error since we have not specified datetime.
#


#---------------------------------------------------------------------------------------------------
# Section 5
#---------------------------------------------------------------------------------------------------
#In this section we will use fi __name__ == "__main__"
#For this we have to use import a function called import_me.py
#If u run the file in the terminal, u will get Hello!!!


from import_me import call_me
#whenn we run this, we get call_me called even though we never made any official call to this function.
#when we import this function, we accidentally execute this module.
#To search for call_me, we had to execute the import_me.py module.
#the main purpose of "if __name__== '__main__' " is to stop this automatic execution from happening.
#Thus if we add the above condition inside the import_me.py file, then we wont get this issue.

from import_me import test
#Notice that since test is not called, inside the module import_me, we dont get "HEll No as output"




























