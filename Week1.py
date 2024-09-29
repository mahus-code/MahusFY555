import random as rand


# Exercise 1
# Make a function that computes the arithmetric function

def arihmetric_series(a0, n, d):
    a_n = a0
    for i in range(1, n+1):
        a_n = a0 + (n-1)*d

    return a_n

# print(arihmetric_series(-5, 5, 3))


def tyr(a0, n, d):
    temp = a0
    for i in range(1, n+1):
        a_n = temp + (i-1)*d
        temp = a_n


    return a_n

# print(tyr(-5, 3, 3))

# --------------------------------------------- Exercise 2 ------------------------------------------------

# Function to generate list of compliments


def compliments():
    n = int(input("How many compliments do you want? "))
    print(n)
    a = []    
    for i in range(n):
        b = input("Write compliment numer %d: " % i)
        a.append(b)
    return a

def randomCompliment(list):
    i = rand.randint(0, len(list)-1)
    print("Your random compliment is: %s" % list[i])

list_of_compliments = compliments()
randomCompliment(list_of_compliments)

