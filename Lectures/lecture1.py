print("Hello World!")
# Comments
def fib(N):
    f0,f1 = 0,1
    f = 0
    if N == 1:
        f = 0
    elif N == 2:
        f = 1
    else:
        for n in range(3, N+1):
            f = f0 + f1
            f0,f1 = f1,f
    return f

N = 19

print("Fibonacci sequence number", N, "equal to, ", fib(N))

