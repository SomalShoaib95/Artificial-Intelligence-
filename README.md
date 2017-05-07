# Artificial-Intelligence-this contain all the labs i've performed in my course.
LAB 1
1.	Write a small program in Python to print your CV.

print ("CV");
print ("Name: KHAAN \t")
print ("GPA: 3.8 \t")
print ("Degree: B(Cs) \t")
print ("Institude: Iqra University")

2.	Write a program that takes the month (1…12) as input. Print whether the season is summer, winter, spring or autumn depending upon the input month.

x = int(input ("Enter a month"))
z= x

if x<=3:
    print ("This is Winter Season!")
elif x<=6:
    print ("This is Summer Season")
elif x<=9:
    print ("This is Summer seasoN")
else:
    print ("This is Autum Season")

3.	To determine whether a year is a leap year

x = int(input("Enter a year: "))

if x % 4 == 0 and x % 100 != 0 or x % 400 == 0:

    print("\n Is a leap-year")

else:

    print("\n Is not a leap-year")

4.	Write a program that takes a line as input and finds the number of letters and digits in the input

x = input ("Type a sentence:\n")

y=z=0

for s in x:

    if s.isdigit():
        y=y+1
    elif s.isalpha():
        z=z+1


print("No. Of Digits are", y)
print ("No. Of Letters are", z)
